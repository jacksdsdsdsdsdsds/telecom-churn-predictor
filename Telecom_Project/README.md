# Telecom Customer Churn Predictor

A production-ready Flask web application that predicts telecom customer churn probability using a Random Forest classifier trained on the IBM Telco Customer Churn dataset.

## Features

- Random Forest churn prediction model trained from the provided Kaggle/IBM Excel dataset
- Flask API with a `POST /predict` endpoint that accepts JSON and returns churn probability
- Single-file dark frontend with a responsive UI
- Plain-language explanation of the most important churn drivers
- Saved model artifact using `joblib`
- Health endpoint for deployment checks

## Project Structure

```text
.
├── app.py
├── artifacts/
│   ├── churn_model.pkl
│   └── metrics.json
├── requirements.txt
├── templates/
│   └── index.html
└── train_model.py
```

## Dataset

This project uses the IBM Telco Customer Churn dataset provided in:

`/Users/yashvenderposwal/Downloads/Telco_customer_churn.xlsx`

The workbook includes the fields used for model training:

- `Tenure Months`
- `Monthly Charges`
- `Phone Service`
- `Multiple Lines`
- `Contract`
- `Tech Support`
- `Payment Method`
- `Churn Value`

## Important Note About Support Calls

The source workbook does **not** contain a native `support calls` column. To preserve the requested product experience:

- The machine learning model is trained only on real dataset-backed features.
- The API still accepts `support_calls`.
- The final churn probability is calibrated with a documented post-model adjustment based on support-call count.

This keeps the model honest to the source data while supporting the requested input form.

## Setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Start the app:

```bash
python app.py
```

The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

For production serving:

```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## API Usage

### `POST /predict`

Request body:

```json
{
  "tenure_months": 8,
  "monthly_charges": 92.5,
  "number_of_lines": 2,
  "contract_type": "Month-to-month",
  "has_tech_support": "No",
  "support_calls": 3,
  "payment_method": "Electronic check"
}
```

Response body:

```json
{
  "risk_level": "HIGH CHURN RISK",
  "churn_probability": 0.7821,
  "confidence_percentage": 78.21,
  "base_model_probability": 0.7014,
  "top_reasons": [
    "Month-to-month contracts historically churn more often than fixed terms.",
    "Short customer tenure is one of the strongest early-churn indicators.",
    "Frequent support contact suggests active service pain points."
  ]
}
```

## Training Notes

- Model type: `RandomForestClassifier`
- Train/test split: `80/20`
- Stratified sampling: enabled
- Class balancing: enabled via `class_weight="balanced"`
- Random seed: `42`

Training metrics are written to `artifacts/metrics.json`.

## Deployment Notes

- `GET /health` returns application and model readiness details
- `gunicorn` is included for production deployment
- Input validation is enforced on the API layer
- Unknown categories are tolerated by the encoder with `handle_unknown="ignore"`
