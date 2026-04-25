from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path("/Users/yashvenderposwal/Downloads/Telco_customer_churn.xlsx")
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "Tenure Months",
    "Monthly Charges",
    "Phone Service",
    "Multiple Lines",
    "Contract",
    "Tech Support",
    "Payment Method",
]
TARGET_COLUMN = "Churn Value"


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    data = pd.read_excel(DATASET_PATH)
    cleaned = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    cleaned["Monthly Charges"] = pd.to_numeric(cleaned["Monthly Charges"], errors="coerce")
    cleaned["Tenure Months"] = pd.to_numeric(cleaned["Tenure Months"], errors="coerce")
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="raise").astype(int)
    return cleaned


def build_pipeline() -> Pipeline:
    numeric_features = ["Tenure Months", "Monthly Charges"]
    categorical_features = ["Phone Service", "Multiple Lines", "Contract", "Tech Support", "Payment Method"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    dataset = load_dataset()
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURE_COLUMNS,
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }

    artifact = {
        "model": pipeline,
        "feature_names": FEATURE_COLUMNS,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "train_rows": metrics["train_rows"],
            "test_rows": metrics["test_rows"],
        },
        "notes": {
            "dataset_path": str(DATASET_PATH),
            "support_calls_note": (
                "The source workbook does not include a support-calls feature. "
                "The API accepts support_calls and applies a documented post-model calibration."
            ),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model artifact to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    main()
