import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def predict(data_path: str, models_dir: str):
    """
    Make predictions using Random Forest model
    """
    # Load data
    df = pd.read_csv(data_path)
    y = df['churn'] if 'churn' in df.columns else None
    X = df.drop('churn', axis=1) if 'churn' in df.columns else df
    
    # Load vectorizer and model
    with open(f"{models_dir}/dict_vectorizer.pkl", 'rb') as f:
        dv = pickle.load(f)
    
    with open(f"{models_dir}/random_forest.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Transform and predict
    X_dict = X.to_dict(orient='records')
    X_vectorized = dv.transform(X_dict)
    
    y_pred = model.predict(X_vectorized)
    y_pred_proba = model.predict_proba(X_vectorized)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    c_report = classification_report(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    metrics = {
        'accuracy': accuracy,
        'classification_report': c_report,
        'roc_auc': roc_auc
    }
    
    return metrics


def predict_single_record(record: dict, models_dir: str) -> dict:
    """
    Make prediction for a single record using Random Forest model
    
    Args:
        record: Dictionary representing a single data record
        models_dir: Directory where the model and vectorizer are stored
    Returns:
        Dictionary with prediction and probability
    """
    # Load vectorizer and model
    with open(f"{models_dir}/dict_vectorizer.pkl", 'rb') as f:
        dv = pickle.load(f)
    
    with open(f"{models_dir}/random_forest.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Transform and predict
    X_vectorized = dv.transform([record])
    
    prediction = model.predict(X_vectorized)[0]
    prediction_proba = model.predict_proba(X_vectorized)[0, 1]
    
    return {
        'prediction': int(prediction),
        'prediction_proba': float(prediction_proba)
    }