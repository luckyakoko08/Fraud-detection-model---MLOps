import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model pipeline
MODEL_PATH = "fraud_detection_pipeline.pkl"
try:
    pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

THRESHOLD = 0.5838

class TransactionInput(BaseModel):
  transaction_id: str
  timestamp: str
  customer_id: str
  card_id: str
  device_id: str
  ip_address: str
  merchant_id: str
  merchant_category: str
  merchant_country: str
  merchant_city: str
  merchant_latitude: float
  merchant_longitude: float
  transaction_type: str
  amount: float
  currency: str
  hour: int
  day_of_week: int
  is_weekend: int
  log_amount: float
  customer_avg_amount: float
  customer_median_amount: float
  amount_to_customer_avg: float
  customer_transaction_count: int
  customer_by_country: int
  customer_device_count: int
  card_transaction_count: int
  card_avg_amount: float
  amount_to_card_avg: float
  device_customer_count: int
  ip_customer_count: int
  ip_transaction_count: int

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": True, "threshold": THRESHOLD}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    try:
        input_data = pd.DataFrame([transaction.dict()])
        probability = float(pipeline.predict_proba(input_data)[:, 1][0])
        is_flagged = probability >= THRESHOLD
        return {
            "fraud_probability": round(probability, 4),
            "decision_threshold": THRESHOLD,
            "flagged_for_review": is_flagged,
            "action": "BLOCK / REVIEW" if is_flagged else "APPROVE"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
