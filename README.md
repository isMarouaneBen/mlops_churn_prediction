# 🎯 MLOps Churn Prediction

A machine learning pipeline for predicting customer churn in e-commerce. Includes data processing, model training with MLflow tracking, REST API, and Docker support.

## 📦 Installation

1. **Clone the repo:**
```bash
git clone https://github.com/isMarouaneBen/mlops_churn_prediction.git
cd mlops_churn_prediction
```

2. **Create virtual environment:**
```powershell
# Windows
python -m venv env
.\env\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Run the Pipeline
```bash
python main.py
```

This ingests data, cleans it, trains models, and logs metrics to MLflow.

### Start the API
```bash
uvicorn api:app --reload
```

Then open http://localhost:8000/docs for interactive API docs.

### View MLflow Dashboard
```bash
mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000 to see experiment results.

## 📁 Project Structure

```
├── main.py                    # Pipeline orchestrator
├── api.py                     # FastAPI inference server
├── config.yaml                # Model hyperparameters
├── steps/                     # Pipeline steps
│   ├── ingest.py             # Load Excel → CSV
│   ├── clean.py              # Clean & preprocess data
│   ├── train.py              # Train models
│   └── predict.py            # Make predictions
├── data/training/
│   ├── raw/                  # Raw data
│   └── processed/            # Cleaned data
├── models/                    # Trained models (.pkl)
└── tests/                     # Unit tests
```

## 🤖 Models

Three classifiers are trained:

| Model | Type | Best For |
|-------|------|----------|
| Logistic Regression | Linear | Baseline, interpretability |
| Decision Tree | Tree-based | Feature importance |
| Random Forest | Ensemble | **Best accuracy** |

## 🌐 API Usage

**Predict churn:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CustomerID": "C001",
    "Tenure": 5,
    "Gender": "Male",
    "SatisfactionScore": 4,
    "OrderCount": 5,
    "DaySinceLastOrder": 30,
    ...
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "churn_probability": 0.18,
  "churn": "No"
}
```

## 🧪 Testing

```bash
pytest tests/
```

## 🐳 Docker

**Build and run:**
```bash
docker build -t churn-prediction .
docker run -p 8000:80 churn-prediction
```

## ⚙️ Configuration

Edit `config.yaml` to adjust model hyperparameters:

```yaml
models:
  logistic_regression:
    params:
      max_iter: 10000
      C: 1.0
  decision_tree:
    params:
      max_depth: 10
  random_forest:
    params:
      n_estimators: 100
      n_jobs: -1
```

## 📊 Pipeline Steps

1. **Ingest** - Load Excel file → CSV
2. **Clean** - Handle missing values, standardize columns/strings
3. **Train** - Train 3 models, log metrics to MLflow
4. **Predict** - Generate predictions with probabilities

## 🔧 Troubleshooting

**Import error?**
```bash
pip install -r requirements.txt
```

**Port 8000 in use?**
```bash
uvicorn api:app --port 8001
```

**Missing data file?**
Ensure `E Commerce Dataset.xlsx` exists in the data folder.

## 📚 Dependencies

- pandas, numpy - Data processing
- scikit-learn - ML models
- mlflow - Experiment tracking
- fastapi, uvicorn - API framework
- pytest - Testing

## 👨‍💻 Author

**Marouane Ben** - [@isMarouaneBen](https://github.com/isMarouaneBen)

## 📄 License

MIT License