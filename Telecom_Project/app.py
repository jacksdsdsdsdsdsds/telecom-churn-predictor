from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "churn_model.pkl"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger("telco-churn-app")


class PredictionError(ValueError):
    pass


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run train_model.py before starting the API."
        )
    artifact = joblib.load(path)
    required_keys = {"model", "metrics", "feature_names"}
    missing = required_keys.difference(artifact.keys())
    if missing:
        raise RuntimeError(f"Model artifact is missing required keys: {sorted(missing)}")
    return artifact


ARTIFACT = load_artifact(MODEL_PATH)
MODEL = ARTIFACT["model"]
MODEL_METRICS = ARTIFACT["metrics"]
TRAINING_NOTES = ARTIFACT.get("notes", {})


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def index() -> str:
        return render_template("index.html", model_metrics=MODEL_METRICS)

    @app.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "model_ready": True,
                "metrics": MODEL_METRICS,
                "notes": TRAINING_NOTES,
            }
        )

    @app.post("/predict")
    def predict() -> Any:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        try:
            normalized = normalize_payload(payload)
            model_features = build_model_features(normalized)
            base_probability = float(MODEL.predict_proba(model_features)[0][1])
            adjusted_probability = adjust_probability(base_probability, normalized)
            risk_level = "HIGH CHURN RISK" if adjusted_probability >= 0.5 else "LOW CHURN RISK"

            return jsonify(
                {
                    "risk_level": risk_level,
                    "churn_probability": round(adjusted_probability, 4),
                    "confidence_percentage": round(max(adjusted_probability, 1 - adjusted_probability) * 100, 2),
                    "base_model_probability": round(base_probability, 4),
                    "top_reasons": explain_prediction(normalized, adjusted_probability),
                    "model_metrics": MODEL_METRICS,
                }
            )
        except PredictionError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            LOGGER.exception("Unexpected prediction failure")
            return jsonify({"error": "Prediction failed due to an internal server error."}), 500

    return app


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    contract_options = {
        "month-to-month": "Month-to-month",
        "one year": "One year",
        "two year": "Two year",
        "three year": "Three year",
        "three years or more": "Three years or more",
    }
    payment_options = {
        "electronic check": "Electronic check",
        "mailed check": "Mailed check",
        "bank transfer": "Bank transfer",
        "credit card": "Credit card",
    }

    try:
        tenure_months = int(payload["tenure_months"])
        monthly_charges = float(payload["monthly_charges"])
        number_of_lines = int(payload["number_of_lines"])
        support_calls = int(payload["support_calls"])
    except KeyError as exc:
        raise PredictionError(f"Missing required field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise PredictionError("Tenure, monthly charges, number of lines, and support calls must be numeric.") from exc

    contract_key = str(payload.get("contract_type", "")).strip().lower()
    payment_key = str(payload.get("payment_method", "")).strip().lower()
    tech_support_raw = str(payload.get("has_tech_support", "")).strip().lower()

    if not 0 <= tenure_months <= 120:
        raise PredictionError("tenure_months must be between 0 and 120.")
    if not 0 <= monthly_charges <= 500:
        raise PredictionError("monthly_charges must be between 0 and 500.")
    if not 0 <= number_of_lines <= 5:
        raise PredictionError("number_of_lines must be between 0 and 5, where 5 represents 5 or more.")
    if not 0 <= support_calls <= 20:
        raise PredictionError("support_calls must be between 0 and 20.")
    if contract_key not in contract_options:
        raise PredictionError(
            "contract_type must be Month-to-month, One year, Two year, Three year, or Three years or more."
        )
    if payment_key not in payment_options:
        raise PredictionError(
            "payment_method must be Electronic check, Mailed check, Bank transfer, or Credit card."
        )
    if tech_support_raw not in {"yes", "no"}:
        raise PredictionError("has_tech_support must be Yes or No.")

    return {
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "number_of_lines": number_of_lines,
        "contract_type": contract_options[contract_key],
        "has_tech_support": tech_support_raw == "yes",
        "support_calls": support_calls,
        "payment_method": payment_options[payment_key],
    }


def build_model_features(normalized: dict[str, Any]) -> pd.DataFrame:
    number_of_lines = normalized["number_of_lines"]
    has_phone_service = number_of_lines > 0
    multiple_lines = "No phone service"
    if number_of_lines == 1:
        multiple_lines = "No"
    elif number_of_lines >= 2:
        multiple_lines = "Yes"

    payment_method = normalized["payment_method"]
    contract_type = normalized["contract_type"]
    payment_mapping = {
        "Electronic check": "Electronic check",
        "Mailed check": "Mailed check",
        "Bank transfer": "Bank transfer (automatic)",
        "Credit card": "Credit card (automatic)",
    }
    contract_mapping = {
        "Month-to-month": "Month-to-month",
        "One year": "One year",
        "Two year": "Two year",
        "Three year": "Two year",
        "Three years or more": "Two year",
    }

    return pd.DataFrame(
        [
            {
                "Tenure Months": normalized["tenure_months"],
                "Monthly Charges": normalized["monthly_charges"],
                "Phone Service": "Yes" if has_phone_service else "No",
                "Multiple Lines": multiple_lines,
                "Contract": contract_mapping[contract_type],
                "Tech Support": "Yes" if normalized["has_tech_support"] else "No",
                "Payment Method": payment_mapping[payment_method],
            }
        ]
    )


def adjust_probability(base_probability: float, normalized: dict[str, Any]) -> float:
    probability = base_probability

    support_calls = normalized["support_calls"]
    if support_calls >= 4:
        probability += 0.12
    elif support_calls == 3:
        probability += 0.08
    elif support_calls == 2:
        probability += 0.04
    elif support_calls == 0:
        probability -= 0.02

    if support_calls >= 3 and not normalized["has_tech_support"]:
        probability += 0.03

    if normalized["tenure_months"] >= 48 and normalized["contract_type"] != "Month-to-month":
        probability -= 0.03

    return float(np.clip(probability, 0.01, 0.99))


def explain_prediction(normalized: dict[str, Any], probability: float) -> list[str]:
    positive_signals: list[tuple[float, str]] = []
    protective_signals: list[tuple[float, str]] = []

    if normalized["contract_type"] == "Month-to-month":
        positive_signals.append((0.95, "Month-to-month contracts historically churn more often than fixed terms."))
    elif normalized["contract_type"] in {"Three year", "Three years or more"}:
        protective_signals.append((0.98, "Long fixed-term contracts are a strong retention signal for this profile."))
    elif normalized["contract_type"] == "Two year":
        protective_signals.append((0.95, "A two-year contract is a strong retention signal in the training data."))
    elif normalized["contract_type"] == "One year":
        protective_signals.append((0.7, "A one-year contract usually lowers churn risk compared with month-to-month plans."))

    if normalized["tenure_months"] <= 12:
        positive_signals.append((0.9, "Short customer tenure is one of the strongest early-churn indicators."))
    elif normalized["tenure_months"] >= 48:
        protective_signals.append((0.85, "Long tenure suggests the customer is already well established."))

    if normalized["monthly_charges"] >= 85:
        positive_signals.append((0.75, "Higher monthly charges tend to correlate with elevated churn risk."))
    elif normalized["monthly_charges"] <= 40:
        protective_signals.append((0.45, "Lower monthly charges generally reduce price-related churn pressure."))

    if normalized["payment_method"] == "Electronic check":
        positive_signals.append((0.6, "Electronic check customers churn more often in the historical telco data."))
    else:
        protective_signals.append((0.35, "Automatic or offline payment methods are usually steadier than electronic checks."))

    if not normalized["has_tech_support"]:
        positive_signals.append((0.65, "Not having tech support increases the chance of unresolved service friction."))
    else:
        protective_signals.append((0.55, "Tech support is a stabilizing signal for customer retention."))

    if normalized["support_calls"] >= 3:
        positive_signals.append((0.8, "Frequent support contact suggests active service pain points."))
    elif normalized["support_calls"] == 0:
        protective_signals.append((0.4, "No recent support calls suggests a smoother service experience."))

    if normalized["number_of_lines"] >= 2 and normalized["monthly_charges"] >= 85:
        positive_signals.append((0.3, "Multiple lines combined with high charges can increase bill sensitivity."))

    signals = positive_signals if probability >= 0.5 else protective_signals
    if not signals:
        signals = positive_signals + protective_signals

    ranked = [message for _, message in sorted(signals, key=lambda item: item[0], reverse=True)]
    return ranked[:3]


app = create_app()


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))
