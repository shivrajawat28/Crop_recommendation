from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "crop_ann_model.joblib"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def train_and_save_model() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "ann",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8, 4),
                    activation="relu",
                    solver="adam",
                    batch_size=10,
                    learning_rate_init=0.001,
                    max_iter=1200,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=35,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    artifact = {
        "model": model,
        "encoder": encoder,
        "features": FEATURES,
        "accuracy": float(accuracy),
        "classes": encoder.classes_.tolist(),
        "feature_ranges": {
            feature: {
                "min": float(df[feature].min()),
                "max": float(df[feature].max()),
                "mean": float(df[feature].mean()),
            }
            for feature in FEATURES
        },
    }

    joblib.dump(artifact, MODEL_PATH)
    return artifact


if __name__ == "__main__":
    saved = train_and_save_model()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Test accuracy: {saved['accuracy']:.4f}")
