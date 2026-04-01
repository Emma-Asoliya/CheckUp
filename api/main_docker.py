"""
main.py — CheckUp FastAPI Backend
===================================
Endpoints:
  GET  /         — root health ping
  GET  /health   — uptime + model status (used by dashboard)
  GET  /metrics  — evaluation metrics (used by dashboard)
  POST /predict  — single .wav file -> mental health prediction
  POST /upload   — multiple .wav files -> saved to data/uploads/
  POST /retrain  — triggers fine-tuning on uploaded files
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
import io
import os
from datetime import datetime

# Import the retraining logic from retrain.py in the same src/ directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retrain import run_retraining

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CheckUp API",
    description="Workplace Mental Health Detection from Audio",
    version="1.0.0"
)

# CORS — allows the frontend (index.html from any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DATA_DIR    = os.path.join(BASE_DIR, 'data', 'uploads')

os.makedirs(DATA_DIR, exist_ok=True)

# ── Load Model & Scaler (once at startup) ─────────────────────────────────────
# Loading on every request would be ~3s per call — we load once and reuse.
# After retraining, the global `model` variable is reloaded in /retrain.
print("Loading model...")
model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'checkup_model.h5'))
mean  = np.load(os.path.join(MODELS_DIR, 'scaler_mean.npy'))
std   = np.load(os.path.join(MODELS_DIR, 'scaler_std.npy'))
label_classes = np.load(
    os.path.join(MODELS_DIR, 'label_classes.npy'),
    allow_pickle=True
)
print(f"Model loaded! Classes: {label_classes}")

# Track server start time for uptime calculation in /health
START_TIME = datetime.now()

# Lock to prevent two simultaneous retrains from corrupting the model file
_retraining_in_progress = False


# ── Helper: MFCC Extraction ───────────────────────────────────────────────────
def extract_mfcc(audio_bytes, n_mfcc=40, max_len=174):
    """
    Converts raw .wav bytes to a normalised MFCC matrix.

    Matches extract_mfcc() in notebook Cell 6 exactly:
      - sr=22050, n_mfcc=40, max_len=174
      - librosa.util.normalize applied
      - zero-padded or truncated to max_len frames
    """
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.normalize(mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint — confirms the API is running."""
    return {
        "message": "CheckUp API is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """
    Returns model uptime and status.
    Called by the frontend every 30s to update the nav status pill
    and the Dashboard uptime badge.

    status is "retraining" while a retrain job is running,
    "healthy" otherwise.
    """
    uptime = datetime.now() - START_TIME
    return {
        "status":         "retraining" if _retraining_in_progress else "healthy",
        "uptime_seconds": uptime.seconds,
        "uptime_human":   str(uptime).split('.')[0],
        "model_loaded":   model is not None,
        "classes":        label_classes.tolist(),
        "model_accuracy": 0.8055,
        "roc_auc":        0.9254
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a single .wav file and returns a workplace mental health prediction.

    Pipeline (mirrors notebook Cells 6-7):
      1. Read uploaded bytes
      2. extract_mfcc() -> (40, 174) matrix
      3. Flatten, apply saved scaler, reshape to (1, 40, 174, 1)
      4. model.predict() -> softmax probabilities
      5. argmax -> label string

    Returns prediction, confidence %, human-readable message,
    and probabilities for all three classes.
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="Only .wav files are supported"
        )

    try:
        audio_bytes = await file.read()
        mfcc        = extract_mfcc(audio_bytes)

        # Scale using original training scaler (notebook Cell 10)
        mfcc_flat   = mfcc.reshape(1, -1)
        mfcc_scaled = ((mfcc_flat - mean) / std).reshape(1, 40, 174, 1)

        prediction      = model.predict(mfcc_scaled, verbose=0)
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

    Files are stored server-side so POST /retrain can find them.
    Non-.wav files are silently skipped and reported in 'failed'.
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
    Triggers fine-tuning of checkup_model.h5 on files in data/uploads/.

    Delegates all work to run_retraining() in retrain.py:
      1. Scan data/uploads/ for .wav files
      2. Parse CREMA-D filenames to derive labels
      3. Extract MFCCs with 4x augmentation
      4. Load checkup_model.h5 as pre-trained base
      5. Freeze blocks 1+2, fine-tune block 3 + dense layers
      6. Save updated model to checkup_model.h5

    After retrain.py finishes, the global `model` variable is reloaded
    so subsequent /predict calls use the new weights immediately —
    no server restart needed.

    Returns files_used, epochs trained, and final validation accuracy.
    Raises 409 if a retrain is already running.
    Raises 400 for expected errors (no files, bad format).
    Raises 500 for unexpected errors.
    """
    global model, _retraining_in_progress

    if _retraining_in_progress:
        raise HTTPException(
            status_code=409,
            detail="A retraining job is already running. Please wait for it to finish."
        )

    _retraining_in_progress = True

    try:
        # All retraining logic lives in retrain.py
        result = run_retraining()

        # Reload updated weights into the global model variable
        # so /predict immediately uses the fine-tuned model
        model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, 'checkup_model.h5')
        )
        print("Global model reloaded with fine-tuned weights.")

        return {
            "message":        "Retraining complete. Model updated successfully.",
            "files_used":     result["files_used"],
            "epochs":         result["epochs"],
            "final_accuracy": result["final_accuracy"]
        }

    except RuntimeError as e:
        # Expected failures: no files, bad filenames, missing model file
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected failures: TF crash, disk full, etc.
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

    finally:
        # Always release the lock, even if an exception was raised
        _retraining_in_progress = False


@app.get("/metrics")
def get_metrics():
    """
    Returns model evaluation metrics for the Dashboard.

    Values are from notebook Cell 14 (sklearn classification_report
    and roc_auc_score on the held-out test set).
    Hardcoded because they represent a fixed evaluation snapshot —
    they update when the model is retrained and re-evaluated in the notebook.
    """
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