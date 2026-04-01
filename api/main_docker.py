"""
main_docker.py — CheckUp FastAPI Backend (TFLite version)
==========================================================
Uses TFLite instead of full TensorFlow to run on Render's free tier.
TFLite model is 3.4MB vs 40.5MB — fits easily in 512MB RAM.

Endpoints:
  GET  /         — serves index.html (the frontend UI)
  GET  /health   — uptime + model status
  GET  /metrics  — evaluation metrics
  POST /predict  — single .wav file -> mental health prediction
  POST /upload   — multiple .wav files -> saved to data/uploads/
  POST /retrain  — triggers fine-tuning on uploaded files
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import librosa
import io
import os
from datetime import datetime

# ── TFLite Runtime ────────────────────────────────────────────────────────────
# Try tflite-runtime first (lightweight, for deployment).
# Fall back to tensorflow.lite if running locally with full TensorFlow installed.
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite-runtime")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("Using tensorflow.lite (fallback)")

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CheckUp API",
    description="Workplace Mental Health Detection from Audio",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
DATA_DIR     = os.path.join(BASE_DIR, 'data', 'uploads')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

os.makedirs(DATA_DIR, exist_ok=True)

# ── Serve Frontend ────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── Load TFLite Model ─────────────────────────────────────────────────────────
# TFLite uses an Interpreter instead of model.predict()
# We load it once at startup and reuse it for every prediction request
print("Loading TFLite model...")
interpreter = tflite.Interpreter(
    model_path=os.path.join(MODELS_DIR, 'checkup_model.tflite')
)
interpreter.allocate_tensors()

# Get input and output tensor details
# These tell us the exact shape and index to use when running inference
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape:  {input_details[0]['shape']}")   # should be (1, 40, 174, 1)
print(f"Output shape: {output_details[0]['shape']}")  # should be (1, 3)

# Load scaler and label classes saved from the original training run
mean          = np.load(os.path.join(MODELS_DIR, 'scaler_mean.npy'))
std           = np.load(os.path.join(MODELS_DIR, 'scaler_std.npy'))
label_classes = np.load(
    os.path.join(MODELS_DIR, 'label_classes.npy'),
    allow_pickle=True
)
print(f"Model loaded! Classes: {label_classes}")

START_TIME = datetime.now()
_retraining_in_progress = False


# ── Helper: MFCC Extraction ───────────────────────────────────────────────────
def extract_mfcc(audio_bytes, n_mfcc=40, max_len=174):
    """
    Converts raw .wav bytes to a normalised MFCC matrix.
    Identical to the notebook Cell 6 implementation.
    """
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.normalize(mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


def run_tflite_inference(mfcc_scaled):
    """
    Runs inference using the TFLite interpreter.

    Unlike model.predict(), TFLite requires:
      1. Setting the input tensor manually
      2. Calling invoke() to run the model
      3. Reading the output tensor manually

    Input must be float32 — TFLite is strict about dtypes.
    """
    # Set the input tensor — must be float32
    interpreter.set_tensor(
        input_details[0]['index'],
        mfcc_scaled.astype(np.float32)
    )

    # Run inference
    interpreter.invoke()

    # Read the output tensor — shape (1, 3) softmax probabilities
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serves index.html at the root URL."""
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))


@app.get("/health")
def health_check():
    """Returns model uptime and status."""
    uptime = datetime.now() - START_TIME
    return {
        "status":         "retraining" if _retraining_in_progress else "healthy",
        "uptime_seconds": uptime.seconds,
        "uptime_human":   str(uptime).split('.')[0],
        "model_loaded":   interpreter is not None,
        "classes":        label_classes.tolist(),
        "model_accuracy": 0.8055,
        "roc_auc":        0.9254
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a single .wav file and returns a workplace mental health prediction.
    Uses TFLite interpreter instead of model.predict() for low memory usage.
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="Only .wav files are supported"
        )

    try:
        audio_bytes = await file.read()

        # Extract and scale MFCC features
        mfcc        = extract_mfcc(audio_bytes)
        mfcc_flat   = mfcc.reshape(1, -1)
        mfcc_scaled = ((mfcc_flat - mean) / std).reshape(1, 40, 174, 1)

        # Run TFLite inference
        prediction      = run_tflite_inference(mfcc_scaled)
        predicted_class = np.argmax(prediction[0])
        confidence      = float(prediction[0][predicted_class])
        label           = label_classes[predicted_class]

        messages = {
            'calm':       'This person appears calm and composed.',
            'stressed':   'Signs of workplace stress detected.',
            'distressed': 'Signs of distress detected. Support recommended.'
        }

        return {
            "prediction":  label,
            "confidence":  round(confidence * 100, 2),
            "message":     messages[label],
            "probabilities": {
                label_classes[i]: round(float(prediction[0][i]) * 100, 2)
                for i in range(len(label_classes))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Accepts multiple .wav files and saves them to data/uploads/.
    """
    saved  = []
    failed = []

    for file in files:
        if not file.filename.endswith('.wav'):
            failed.append(file.filename)
            continue
        try:
            save_path = os.path.join(DATA_DIR, file.filename)
            with open(save_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            saved.append(file.filename)
        except Exception:
            failed.append(file.filename)

    return {
        "message":          f"Successfully uploaded {len(saved)} files",
        "saved":            saved,
        "failed":           failed,
        "total_uploaded":   len(saved),
        "upload_directory": DATA_DIR
    }


@app.post("/retrain")
async def retrain_model():
    """
    Triggers fine-tuning on uploaded files.
    Note: retraining still uses full TensorFlow (runs in the notebook/locally).
    After retraining, convert the new .h5 to .tflite and redeploy.
    """
    global _retraining_in_progress

    if _retraining_in_progress:
        raise HTTPException(
            status_code=409,
            detail="A retraining job is already running. Please wait."
        )

    # Import retrain here to avoid loading TF at startup
    # (retrain.py uses full TensorFlow, not TFLite)
    _retraining_in_progress = True

    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from retrain import run_retraining
        result = run_retraining()

        return {
            "message":        "Retraining complete. Redeploy to use updated model.",
            "files_used":     result["files_used"],
            "epochs":         result["epochs"],
            "final_accuracy": result["final_accuracy"]
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
    finally:
        _retraining_in_progress = False


@app.get("/metrics")
def get_metrics():
    """Returns model evaluation metrics for the Dashboard."""
    return {
        "accuracy":          0.8055,
        "precision":         0.8119,
        "recall":            0.8055,
        "f1_score":          0.8071,
        "roc_auc":           0.9254,
        "dataset":           "CREMA-D",
        "total_samples":     7442,
        "augmented_samples": 29768,
        "classes":           label_classes.tolist()
    }