from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

app = FastAPI()

class CustomerFeatures(BaseModel):
    CustomerID: str
    Tenure: Optional[int] = None
    PreferredLoginDevice: Optional[str] = None
    CityTier: Optional[int] = None
    WarehouseToHome: Optional[float] = None
    PreferredPaymentMode: Optional[str] = None
    Gender: Optional[str] = None
    HourSpendOnApp: Optional[float] = None
    NumberOfDeviceRegistered: Optional[int] = None
    PreferedOrderCat: Optional[str] = None
    SatisfactionScore: Optional[int] = None
    MaritalStatus: Optional[str] = None
    NumberOfAddress: Optional[int] = None
    Complain: Optional[int] = None
    OrderAmountHikeFromlastYear: Optional[float] = None
    CouponUsed: Optional[int] = None
    OrderCount: Optional[int] = None
    DaySinceLastOrder: Optional[int] = None
    CashbackAmount: Optional[float] = None


def predict_single_record(features: dict, models_dir: str = "models"):
    """Load model and vectorizer, make prediction"""
    try:
        with open(f"{models_dir}/dict_vectorizer.pkl", 'rb') as f:
            dv = pickle.load(f)
        with open(f"{models_dir}/random_forest.pkl", 'rb') as f:
            model = pickle.load(f)
        
        X_vectorized = dv.transform([features])
        prediction = model.predict(X_vectorized)[0]
        probability = model.predict_proba(X_vectorized)[0][1]
        
        return {
            "prediction": int(prediction),
            "churn_probability": float(probability),
            "churn": "Yes" if prediction == 1 else "No"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def home():
    return {"message": "FastAPI is working!"}

@app.post("/predict")
async def predict_churn(features: CustomerFeatures):
    prediction = predict_single_record(features.dict(), models_dir="models")
    return prediction