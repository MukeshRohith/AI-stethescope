import argparse
import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape: tuple[int, int, int]) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=r"D:\classification-of-heart-sound-recordings-the-physionetcomputing-in-cardiology-challenge-2016-1.0.0\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0",
    )
    parser.add_argument("--x-name", default="X.npy")
    parser.add_argument("--y-name", default="y.npy")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-model", default=r"d:\ai steth app\physionet_cnn.keras")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    x_path = os.path.join(data_dir, args.x_name)
    y_path = os.path.join(data_dir, args.y_name)

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, 64, 157); got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y with shape (N,); got {y.shape}")

    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)

    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(input_shape=X_train.shape[1:])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Abnormal (1)"]))

    out_model = os.path.abspath(args.out_model)
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    model.save(out_model)
    print(f"Saved model: {out_model}")


if __name__ == "__main__":
    main()

