import os
import io

import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt

MODEL_PATH = os.path.join(os.path.dirname(__file__), "physionet_cnn.keras")
SAMPLE_RATE_HZ = 16000
TARGET_SECONDS = 5
TARGET_SAMPLES = SAMPLE_RATE_HZ * TARGET_SECONDS
TARGET_FRAMES = 157

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/ping")
async def ping():
    return {"ok": True}


def bandpass_filter(data: np.ndarray, fs: int, lowcut: float = 20.0, highcut: float = 200.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def take_middle_or_pad_5s(y: np.ndarray) -> np.ndarray:
    if y.shape[0] >= TARGET_SAMPLES:
        start = (y.shape[0] - TARGET_SAMPLES) // 2
        return y[start : start + TARGET_SAMPLES]
    pad_total = TARGET_SAMPLES - y.shape[0]
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant")


def to_64x157(spec_db: np.ndarray) -> np.ndarray:
    if spec_db.shape[1] == TARGET_FRAMES:
        return spec_db
    if spec_db.shape[1] > TARGET_FRAMES:
        return spec_db[:, :TARGET_FRAMES]
    pad_val = float(np.min(spec_db)) if spec_db.size else -80.0
    pad_width = TARGET_FRAMES - spec_db.shape[1]
    return np.pad(spec_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=pad_val)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        y, sr = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE_HZ, mono=True)
        y = take_middle_or_pad_5s(y)
        y = bandpass_filter(y, sr, lowcut=20.0, highcut=200.0, order=4).astype(np.float32, copy=False)

        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=200.0)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_db = to_64x157(spec_db).astype(np.float32, copy=False)

        x = spec_db[np.newaxis, ..., np.newaxis]
        prob = float(model.predict(x, verbose=0).reshape(-1)[0])
        prediction = 1 if prob > 0.5 else 0
        diagnosis = "ABNORMAL" if prediction == 1 else "NORMAL"
        return {
            "prediction": prediction,
            "diagnosis": diagnosis,
            "probability": prob,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
