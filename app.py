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
    amount: float
    customer_avg_amount: float
    customer_median_amount: float
    customer_device_count: int
    amount_to_customer_avg: float
    card_avg_amount: float
    card_transaction_count: int
    merchant_category: str
    merchant_country: str
    hour: int

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
