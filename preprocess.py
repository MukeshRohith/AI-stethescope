import argparse
import os

import librosa
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm


def bandpass_filter(data: np.ndarray, fs: int, lowcut: float = 20.0, highcut: float = 200.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def load_reference_map(folder_path: str) -> dict[str, int]:
    candidates = ["REFERENCE.csv", "REFERENCE.txt", "REFERENCE"]
    ref_path = None
    for name in candidates:
        p = os.path.join(folder_path, name)
        if os.path.exists(p):
            ref_path = p
            break
    if ref_path is None:
        raise FileNotFoundError(f"REFERENCE file not found in: {folder_path}")

    mapping: dict[str, int] = {}
    with open(ref_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.lower().startswith("record") and ("label" in line.lower() or "," in line):
                continue
            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            if len(parts) < 2:
                continue
            rec_id = os.path.splitext(parts[0])[0]
            try:
                raw_label = int(parts[1])
            except ValueError:
                continue
            if raw_label == 1:
                mapping[rec_id] = 0
            elif raw_label == -1:
                mapping[rec_id] = 1
    return mapping


def pad_or_truncate(y: np.ndarray, target_len: int) -> np.ndarray:
    if y.shape[0] == target_len:
        return y
    if y.shape[0] > target_len:
        return y[:target_len]
    out = np.zeros(target_len, dtype=y.dtype)
    out[: y.shape[0]] = y
    return out


def process_wav(wav_path: str, sr: int, target_len: int, n_mels: int, fmax: float) -> np.ndarray:
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    y = pad_or_truncate(y, target_len)
    y = bandpass_filter(y, sr, lowcut=20.0, highcut=200.0, order=4)
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    s_db = librosa.power_to_db(s, ref=np.max)
    return s_db.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--fmax", type=float, default=200.0)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else root
    os.makedirs(out_dir, exist_ok=True)

    training_folders = [os.path.join(root, f"training-{c}") for c in "abcdef"]
    items: list[tuple[str, int]] = []

    for folder in training_folders:
        if not os.path.isdir(folder):
            continue
        ref = load_reference_map(folder)
        for name in os.listdir(folder):
            if not name.lower().endswith(".wav"):
                continue
            rec_id = os.path.splitext(name)[0]
            if rec_id not in ref:
                continue
            items.append((os.path.join(folder, name), ref[rec_id]))

    if not items:
        raise RuntimeError("No labeled .wav files found in training-a to training-f.")

    target_len = int(args.sr * args.seconds)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for wav_path, label in tqdm(items, desc="Preprocessing", unit="file"):
        spec = process_wav(wav_path, sr=args.sr, target_len=target_len, n_mels=args.n_mels, fmax=args.fmax)
        X_list.append(spec)
        y_list.append(label)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    print(f"Saved X.npy: {X.shape} -> {os.path.join(out_dir, 'X.npy')}")
    print(f"Saved y.npy: {y.shape} -> {os.path.join(out_dir, 'y.npy')}")


if __name__ == "__main__":
    main()

