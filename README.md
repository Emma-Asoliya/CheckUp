# CheckUp — Workplace Mental Health Detection from Audio

> An end-to-end machine learning pipeline that classifies voice recordings into workplace mental health states: **calm**, **stressed**, or **distressed**.


##  Video Demo
[]

## URL
[https://checkup-production.up.railway.app](https://checkup-production.up.railway.app)



## Project Description

CheckUp is a machine learning pipeline that uses the CREMA-D audio dataset. It extracts MFCC (Mel Frequency Cepstral Coefficient) features from voice recordings and classifies them using a 3-block CNN trained in TensorFlow.

The project builds on prior mental health prediction research by moving from tabular survey data to real audio signals — making detection more immediate and applicable in real workplace settings.

### The system supports:
- **Single-file prediction** — drag and drop a .wav file and get an instant mental health classification
- **Bulk upload** — upload multiple new .wav files for retraining
- **Fine-tuning** — retrain the existing model on new data using it as a pre-trained base
- **Dashboard** — live model uptime, evaluation metrics, and dataset visualizations
- **REST API** — fully documented FastAPI backend with Swagger UI

### Dataset
**CREMA-D** — 7,442 audio files from 91 actors expressing 6 emotions (angry, disgust, fearful, happy, neutral, sad). Emotions are mapped to 3 workplace states:

| Emotion | Workplace Label |
|---------|----------------|
| Angry, Disgust, Fearful | Stressed |
| Happy, Neutral | Calm |
| Sad | Distressed |

After data augmentation (noise, pitch shift, time stretch): **29,768 training samples**

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 80.55% |
| Precision | 81.19% |
| Recall | 80.55% |
| F1 Score | 80.71% |
| ROC-AUC | 92.54% |


## Project Structure

```
CheckUp/
├── README.md
├── Dockerfile
├── requirements.txt
├── locustfile.py
├── notebook/
│   └── CheckUp.ipynb           # Full training pipeline
├── api/
│   ├── main_docker.py          # FastAPI backend (all endpoints)
│   └── retrain.py              # Retraining script (standalone + used by API)
├── data/
│   ├── train/                  # Original CREMA-D training data
│   ├── test/                   # Test split
│   └── uploads/                # New files uploaded via UI for retraining
├── models/
│   ├── checkup_model.h5        # Trained CNN model
│   ├── scaler_mean.npy         # Training set mean (for normalization)
│   ├── scaler_std.npy          # Training set std (for normalization)
│   └── label_classes.npy       # Class label array ['calm', 'distressed', 'stressed']
└── frontend/
    └── index.html              # Single-page UI
```


## Setup & Run Instructions

### Prerequisites
- Python 3.11
- pip


###  Run Locally

#### 1. Clone the repository
```bash
git clone https://github.com/Emma-Asoliya/CheckUp.git
cd CheckUp
```

#### 2. Create and activate a virtual environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Confirm your models folder has these files
```
models/
  checkup_model.h5
  scaler_mean.npy
  scaler_std.npy
  label_classes.npy
```

If any are missing, run the notebook (`notebook/CheckUp.ipynb`) end-to-end first.

#### 5. Start the FastAPI server
```bash
uvicorn api.main_docker:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
Loading model...
Model loaded! Classes: ['calm' 'distressed' 'stressed']
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### 6. Open the frontend
Open `frontend/index.html` directly in your browser or visit `http://localhost:8000`

#### 7. Verify everything works
Visit `http://127.0.0.1:8000/docs` to see the interactive Swagger UI.



## CREMA-D File Naming Format

For retraining to work, uploaded files **must** follow the CREMA-D naming convention:

```
ActorID_SentenceID_EmotionCode_Level.wav
```

Examples:
```
1001_DFA_ANG_XX.wav   → angry   → stressed
1002_IEO_HAP_HI.wav   → happy   → calm
1003_ITS_SAD_LO.wav   → sad     → distressed
```

Supported emotion codes:

| Code | Emotion | Workplace Label |
|------|---------|-----------------|
| ANG | Angry | stressed |
| DIS | Disgust | stressed |
| FEA | Fearful | stressed |
| HAP | Happy | calm |
| NEU | Neutral | calm |
| SAD | Sad | distressed |


## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Serves the frontend UI |
| GET | /health | Uptime, model status, class names |
| GET | /metrics | Accuracy, F1, ROC-AUC, sample counts |
| POST | /predict | Upload one .wav → get prediction |
| POST | /upload | Upload multiple .wav files for retraining |
| POST | /retrain | Trigger fine-tuning on uploaded files |

Full interactive docs: `https://checkup-production.up.railway.app/docs`


## Load Testing Results (Locust)

Tested against: `https://checkup-production.up.railway.app`

Tool: [Locust](https://locust.io) — open source load testing framework

### Results

| Users | Requests | Failures | Median (ms) | 95th % (ms) | 99th % (ms) | Avg (ms) | RPS |
|-------|----------|----------|-------------|-------------|-------------|----------|-----|
| 10 | 33 | 0 (0%) | 590 | 1300 | 1400 | 672 | 1.0 |
| 50 | 206 | 0 (0%) | 480 | 850 | 1000 | 511 | 1.04 |
| 100 | 2445 | 0 (0%) | 350 | 500 | 620 | 387 | 11.92 |

### Observations
- **Zero failures** across all three runs — the model handles concurrent requests reliably
- **Response time improves at higher load** — at 100 users the server is fully warmed up and processing requests efficiently
- **12 RPS at 100 users** — sufficient throughput for a workplace deployment

### How to run the load test
```bash
pip install locust
locust -f locustfile.py --host https://checkup-production.up.railway.app
```
Then open `http://localhost:8089` in your browser.


## Model Architecture

3-block CNN trained on 40×174 MFCC matrices:

```
Input (40, 174, 1)
  → Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)   # Block 1
  → Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)   # Block 2
  → Conv2D(128) + BatchNorm + MaxPool + Dropout(0.3)   # Block 3
  → Flatten
  → Dense(256, relu) + Dropout(0.5)
  → Dense(3, softmax)                                   # Output
```

### Optimization Techniques
1. **Data Augmentation** — 4x dataset expansion (noise, pitch shift, time stretch)
2. **Batch Normalization** — stabilizes training across all 3 CNN blocks
3. **Dropout Regularization** — prevents overfitting (0.25 in conv layers, 0.5 in dense)
4. **Class Weights** — compensates for class imbalance
5. **Early Stopping** — restores best weights automatically (patience=10)
6. **ReduceLROnPlateau** — adaptive learning rate (patience=5, factor=0.5)


## Retraining Pipeline

The retraining system fine-tunes the existing model on new uploaded data:

1. Scans `data/uploads/` for `.wav` files
2. Parses CREMA-D filenames to derive labels automatically
3. Runs MFCC extraction with 4x augmentation on each file
4. Loads `checkup_model.h5` as a **pre-trained base**
5. **Freezes blocks 1 & 2** (preserves learned audio features)
6. Fine-tunes **block 3 + dense layers** at lr=0.0001
7. Saves updated weights back to `checkup_model.h5`
8. Reloads the model in memory — no server restart needed


## Tech Stack

| Layer | Technology |
|-------|------------|
| Model | TensorFlow / Keras (CNN) |
| Audio | Librosa (MFCC extraction) |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Railway |
| Load Test | Locust |