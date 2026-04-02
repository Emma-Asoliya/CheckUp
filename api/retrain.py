"""
retrain.py — CheckUp Model Retraining Script
=============================================
This script fine-tunes the existing checkup_model.h5 on newly uploaded
.wav files from the data/uploads/ directory.

It mirrors the training pipeline from the notebook exactly:
  - Same MFCC extraction (n_mfcc=40, max_len=174)
  - Same 4x augmentation (noise, pitch shift, time stretch)
  - Same label mapping (CREMA-D emotion codes → workplace labels)
  - Same scaler (loaded from scaler_mean.npy / scaler_std.npy)

The key difference from training from scratch:
  - The first two CNN blocks are FROZEN (weights preserved)
  - Only the third CNN block + dense layers are fine-tuned
  - Learning rate is 10x lower (0.0001 vs 0.001) to avoid catastrophic forgetting

Usage:
  python src/retrain.py
  (or called automatically by the POST /retrain FastAPI endpoint)
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

# ── Path Setup ────────────────────────────────────────────────────────────────
# Works whether called directly (python src/retrain.py) or imported by main.py
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'data', 'uploads')

# ── Label Maps ────────────────────────────────────────────────────────────────
# Identical to the notebook's EMOTION_MAP + WORKPLACE_MAP (Cell 3).
# CREMA-D filenames follow: ActorID_SentenceID_EmotionCode_Level.wav
# e.g. 1001_DFA_ANG_XX.wav → parts[2] = 'ANG' → 'angry' → 'stressed'
EMOTION_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

WORKPLACE_MAP = {
    'angry':   'stressed',
    'disgust': 'stressed',
    'fearful': 'stressed',
    'happy':   'calm',
    'neutral': 'calm',
    'sad':     'distressed'
}


def parse_label(filename):
    """
    Extracts the workplace label from a CREMA-D formatted filename.

    CREMA-D format: ActorID_SentenceID_EmotionCode_Level.wav
    Example: 1001_DFA_ANG_XX.wav → 'stressed'

    Returns None if the filename does not match the expected format
    or contains an unrecognised emotion code.
    """
    try:
        parts = os.path.splitext(filename)[0].split('_')
        # Emotion code is always the third underscore-separated segment
        emotion_code = parts[2]
        emotion = EMOTION_MAP.get(emotion_code)
        if emotion is None:
            return None
        return WORKPLACE_MAP.get(emotion)
    except (IndexError, AttributeError):
        return None


def extract_mfcc_augmented(file_path, n_mfcc=40, max_len=174):
    """
    Loads a .wav file and returns 4 augmented MFCC matrices.

    Augmentations (identical to extract_mfcc_augmented() in Cell 9 of the notebook):
      1. Original audio
      2. Gaussian noise  (std = 0.005)
      3. Pitch shift     (+2 semitones)
      4. Time stretch    (rate = 0.9 — slightly slower)

    Each MFCC matrix is:
      - Shape: (40, 174)  — n_mfcc rows x max_len time frames
      - Normalised with librosa.util.normalize
      - Padded with zeros if shorter than max_len
      - Truncated if longer than max_len

    Returns a list of 4 numpy arrays, or None on error.
    """
    try:
        # Load at 22050 Hz — same sample rate used during original training
        audio, sr = librosa.load(file_path, sr=22050)

        def get_mfcc(y):
            """Extract, normalize, and pad/truncate one MFCC matrix."""
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc = librosa.util.normalize(mfcc)
            if mfcc.shape[1] < max_len:
                mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
            else:
                mfcc = mfcc[:, :max_len]
            return mfcc

        results = []

        # Augmentation 1: Original (unmodified)
        results.append(get_mfcc(audio))

        # Augmentation 2: Add small Gaussian noise to simulate background noise
        noise = np.random.randn(len(audio)) * 0.005
        results.append(get_mfcc(audio + noise))

        # Augmentation 3: Pitch shift up by 2 semitones
        pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        results.append(get_mfcc(pitched))

        # Augmentation 4: Slow down slightly (time stretch at 0.9x speed)
        stretched = librosa.effects.time_stretch(audio, rate=0.9)
        results.append(get_mfcc(stretched))

        return results

    except Exception as e:
        print(f"  [WARN] Could not process {os.path.basename(file_path)}: {e}")
        return None


def run_retraining():
    """
    Main retraining function. Called by:
      - The POST /retrain FastAPI endpoint (main.py)
      - Directly via: python src/retrain.py

    Returns a dict with:
      {
        "files_used":     int,   # number of .wav files successfully processed
        "epochs":         int,   # actual epochs trained (early stopping may reduce this)
        "final_accuracy": float  # validation accuracy of the last epoch
      }

    Raises RuntimeError if there are no valid files to train on.
    """

    print("\n" + "="*55)
    print("  CheckUp — Model Retraining")
    print("="*55)

    # ── Step 1: Discover uploaded files ──────────────────────────────────────
    print(f"\n[1/5] Scanning uploads directory: {UPLOADS_DIR}")

    if not os.path.exists(UPLOADS_DIR):
        raise RuntimeError(f"Uploads directory not found: {UPLOADS_DIR}")

    wav_files = [f for f in os.listdir(UPLOADS_DIR) if f.endswith('.wav')]

    if len(wav_files) == 0:
        raise RuntimeError(
            "No .wav files found in uploads directory. "
            "Please upload files via the UI before retraining."
        )

    print(f"  Found {len(wav_files)} .wav file(s)")

    # ── Step 2: Extract MFCC features with augmentation ──────────────────────
    # Mirrors Cell 9 of the notebook exactly
    print(f"\n[2/5] Extracting MFCC features (4x augmentation per file)...")

    X, y = [], []
    skipped = 0

    for filename in wav_files:
        # Parse the workplace label from the CREMA-D filename
        label = parse_label(filename)
        if label is None:
            print(f"  [SKIP] {filename} — unrecognised format or emotion code")
            skipped += 1
            continue

        file_path = os.path.join(UPLOADS_DIR, filename)
        augmented = extract_mfcc_augmented(file_path)

        if augmented is None:
            skipped += 1
            continue

        # Each file produces 4 augmented samples all sharing the same label
        for mfcc in augmented:
            X.append(mfcc)
            y.append(label)

    files_used = len(wav_files) - skipped
    print(f"  Processed: {files_used} files → {len(X)} samples (after 4x augmentation)")
    print(f"  Skipped:   {skipped} files (bad format or processing error)")

    if files_used == 0:
        raise RuntimeError(
            "No files could be processed. "
            "Ensure files follow CREMA-D naming: ActorID_SentenceID_EmotionCode_Level.wav"
        )

    # ── Step 3: Prepare data ──────────────────────────────────────────────────
    # Encode labels, reshape, scale — identical to Cell 10 of the notebook
    print(f"\n[3/5] Preparing data...")

    X = np.array(X)
    y = np.array(y)

    # Load saved label classes from original training to keep encoding consistent.
    # This guarantees class 0/1/2 map to the same labels as when the model was built.
    label_classes = np.load(
        os.path.join(MODELS_DIR, 'label_classes.npy'),
        allow_pickle=True
    )
    label_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    y_encoded = np.array([label_to_idx[label] for label in y])

    print(f"  Label mapping (preserved from original training):")
    for lbl, idx in label_to_idx.items():
        count = np.sum(y_encoded == idx)
        print(f"    {lbl} -> {idx}  ({count} samples)")

    # Reshape to (N, 40, 174, 1) — CNN expects a channel dimension
    X_reshaped = X.reshape(X.shape[0], 40, 174, 1)

    # Split into train/val — stratify preserves class balance
    # Fall back to training on everything if the dataset is very small
    if len(X_reshaped) < 10:
        print("  [WARN] Very few samples — training without validation split")
        X_train, X_val = X_reshaped, X_reshaped
        y_train, y_val = y_encoded, y_encoded
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_reshaped, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

    # Scale using the ORIGINAL training scaler (not a new one fitted on uploads).
    # Using the original mean/std ensures the feature space seen by the model
    # matches what it was trained on — a new scaler would shift all values.
    mean = np.load(os.path.join(MODELS_DIR, 'scaler_mean.npy'))
    std  = np.load(os.path.join(MODELS_DIR, 'scaler_std.npy'))

    X_train_flat   = X_train.reshape(X_train.shape[0], -1)
    X_val_flat     = X_val.reshape(X_val.shape[0], -1)
    X_train_scaled = ((X_train_flat - mean) / std).reshape(X_train.shape)
    X_val_scaled   = ((X_val_flat   - mean) / std).reshape(X_val.shape)

    # One-hot encode for categorical_crossentropy
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes=3)

    # Class weights to handle imbalanced uploads
    unique_classes       = np.unique(y_train)
    class_weights_array  = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    class_weight_dict = dict(zip(unique_classes.tolist(), class_weights_array.tolist()))
    print(f"  Class weights:     {class_weight_dict}")
    print(f"  Training samples:  {len(X_train)}")
    print(f"  Validation samples:{len(X_val)}")

    # ── Step 4: Load pre-trained model and configure for fine-tuning ─────────
    print(f"\n[4/5] Loading pre-trained model and configuring for fine-tuning...")

    model_path = os.path.join(MODELS_DIR, 'checkup_model.h5')
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"  Model loaded from: {model_path}")
    print(f"  Total layers: {len(model.layers)}")

    # Freeze strategy:
    #   Layers 0-8  (blocks 1 & 2) → FROZEN   — preserve learned audio features
    #   Layers 9+   (block 3 + dense) → TRAINABLE — adapt to new data
    #
    # Full layer map (notebook Cell 11 architecture):
    #   0  InputLayer
    #   1  Conv2D(32)     Block 1
    #   2  BatchNorm
    #   3  MaxPooling2D
    #   4  Dropout
    #   5  Conv2D(64)     Block 2
    #   6  BatchNorm
    #   7  MaxPooling2D
    #   8  Dropout
    #   9  Conv2D(128)    Block 3  ← fine-tune from here
    #   10 BatchNorm
    #   11 MaxPooling2D
    #   12 Dropout
    #   13 Flatten
    #   14 Dense(256)
    #   15 Dropout
    #   16 Dense(3)
    FREEZE_UP_TO_LAYER = 9

    frozen_count = trainable_count = 0
    for i, layer in enumerate(model.layers):
        if i < FREEZE_UP_TO_LAYER:
            layer.trainable = False
            frozen_count += 1
        else:
            layer.trainable = True
            trainable_count += 1

    print(f"  Frozen layers:    {frozen_count} (blocks 1 & 2 preserved)")
    print(f"  Trainable layers: {trainable_count} (block 3 + dense head)")

    # Lower learning rate prevents overwriting the frozen layers' knowledge
    # (0.0001 = 10x smaller than the original 0.001 used in the notebook)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ── Step 5: Fine-tune ─────────────────────────────────────────────────────
    print(f"\n[5/5] Fine-tuning model...")

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,               # stop if no improvement for 5 epochs
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,               # halve LR on plateau
            patience=3,
            min_lr=0.000001,
            verbose=1
        )
    ]

    # 20 epochs max for fine-tuning — much less than the original 80
    # because we are only updating a fraction of the model's layers
    history = model.fit(
        X_train_scaled, y_train_cat,
        batch_size=32,
        epochs=20,
        validation_data=(X_val_scaled, y_val_cat),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    epochs_trained = len(history.history['accuracy'])
    final_accuracy = float(history.history['val_accuracy'][-1])
    best_accuracy  = float(max(history.history['val_accuracy']))

    print(f"\n  Fine-tuning complete!")
    print(f"  Epochs trained: {epochs_trained}")
    print(f"  Final val acc:  {final_accuracy:.4f}")
    print(f"  Best val acc:   {best_accuracy:.4f}")

    # Overwrite the model file so the API uses the updated weights.
    # The /retrain endpoint in main.py reloads the global model after this returns.
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    print("\n" + "="*55)
    print("  Retraining complete!")
    print("="*55 + "\n")

    return {
        "files_used":     files_used,
        "epochs":         epochs_trained,
        "final_accuracy": final_accuracy
    }


# ── Entry point ───────────────────────────────────────────────────────────────
# Allows running standalone: python src/retrain.py
if __name__ == "__main__":
    try:
        result = run_retraining()
        print(f"Result: {result}")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
