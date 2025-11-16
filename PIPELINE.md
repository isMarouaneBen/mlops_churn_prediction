# Churn Prediction Pipeline with MLflow

A complete machine learning pipeline for customer churn prediction with experiment tracking using MLflow.

## Project Structure

```
├── main.py                          # Main pipeline orchestrator
├── steps/                           # Individual pipeline steps
│   ├── __init__.py
│   ├── 01_ingest.py                # Data ingestion
│   ├── 02_clean.py                 # Data cleaning & preprocessing
│   ├── 03_train.py                 # Model training
│   ├── train_with_mlflow.py        # MLflow-integrated training
│   ├── 04_predict.py               # Predictions
│   └── pipeline.py                 # Step orchestrator
├── data/                            # Data directories
│   ├── raw/                         # Raw ingested data
│   ├── processed/                   # Cleaned data
│   └── external/                    # External datasets
├── models/                          # Trained models
├── predictions/                     # Model predictions
├── mlflow_runs/                     # MLflow tracking data
├── config/                          # Configuration files
├── api/                             # API endpoints
├── tests/                           # Test suite
└── requirements.txt                 # Python dependencies

```

## Pipeline Steps

### Step 1: Data Ingestion
- Loads data from Excel file
- Converts to CSV format
- Saves to `data/raw/`

### Step 2: Data Cleaning
- Handles missing values
- Standardizes column names (lowercase, underscores)
- Normalizes string values
- Saves to `data/processed/`

### Step 3: Model Training with MLflow
Trains three classification models:
1. **Logistic Regression**
   - Max iterations: 10,000
   - Solver: lbfgs
   - Regularization (C): 1.0

2. **Decision Tree Classifier**
   - Max depth: 10
   - Random state: 42

3. **Random Forest Classifier**
   - Number of estimators: 100
   - Parallel jobs: -1 (use all cores)
   - Random state: 42

**Metrics Tracked:**
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

### Step 4: Predictions
- Loads trained models
- Makes predictions on data
- Saves predictions to `predictions/`

## Installation

### 1. Create Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv env
.\env\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline
```powershell
# Using default paths
python main.py

# Using custom paths
python main.py `
    --input-file "./data/E Commerce Dataset.xlsx" `
    --raw-data-dir "./data/raw" `
    --processed-data-dir "./data/processed" `
    --models-dir "./models" `
    --predictions-dir "./predictions" `
    --mlflow-uri "./mlflow_runs" `
    --experiment-name "Churn_Prediction_Experiment"
```

### Run Individual Steps

**Step 1 - Data Ingestion:**
```bash
python steps/01_ingest.py
```

**Step 2 - Data Cleaning:**
```bash
python steps/02_clean.py
```

**Step 3 - Model Training (with MLflow):**
```bash
python steps/train_with_mlflow.py
```

**Step 4 - Predictions:**
```bash
python steps/04_predict.py
```

### View MLflow Dashboard
```bash
mlflow ui --backend-store-uri ./mlflow_runs
```
Then open `http://localhost:5000` in your browser.

## Output Files

### Models
- `models/logistic_regression.pkl` - Trained Logistic Regression model
- `models/decision_tree.pkl` - Trained Decision Tree model
- `models/random_forest.pkl` - Trained Random Forest model
- `models/dict_vectorizer.pkl` - Feature vectorizer (required for predictions)

### Data
- `data/raw/raw_data.csv` - Raw ingested data
- `data/processed/cleaned_data.csv` - Cleaned & processed data

### Predictions
- `predictions/logistic_regression_predictions.csv` - Logistic Regression predictions
- `predictions/decision_tree_predictions.csv` - Decision Tree predictions
- `predictions/random_forest_predictions.csv` - Random Forest predictions

### Logs
- `pipeline.log` - Complete pipeline execution log

## MLflow Features

The pipeline integrates MLflow for:
- **Experiment Tracking**: Group related runs under an experiment
- **Parameter Logging**: Track model hyperparameters
- **Metric Logging**: Log performance metrics for each model
- **Model Registry**: Save trained models for reproducibility
- **Nested Runs**: Each model training is a nested run under the main pipeline run
- **Tags**: Automatically tag runs with model types

## Configuration

### Model Hyperparameters
Edit the hyperparameters in `steps/train_with_mlflow.py`:

```python
# Logistic Regression
LogisticRegression(max_iter=10000, solver='lbfgs', C=1.0)

# Decision Tree
DecisionTreeClassifier(random_state=42, max_depth=10)

# Random Forest
RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

### Pipeline Parameters
Edit default paths in `main.py`:

```python
run_full_pipeline(
    input_file="./data/E Commerce Dataset.xlsx",
    raw_data_dir="./data/raw",
    processed_data_dir="./data/processed",
    models_dir="./models",
    predictions_dir="./predictions",
    mlflow_tracking_uri="./mlflow_runs",
    experiment_name="Churn_Prediction_Experiment"
)
```

## Troubleshooting

### Missing Data File
Ensure the Excel file exists at the specified path:
```bash
data/E Commerce Dataset.xlsx
```

### MLflow Not Found
Install MLflow:
```bash
pip install mlflow
```

### Permission Denied
On Windows, if activation fails:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Out of Memory
For large datasets, reduce the `n_jobs` parameter in Random Forest:
```python
RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
```

## Performance Benchmarks

Typical execution times (on standard hardware):
- Data Ingestion: ~2-5 seconds
- Data Cleaning: ~1-3 seconds
- Model Training: ~30-60 seconds
- Predictions: ~5-10 seconds
- **Total: ~40-80 seconds**

## Next Steps

1. **Model Improvement**
   - Hyperparameter tuning
   - Feature engineering
   - Cross-validation

2. **Deployment**
   - REST API (FastAPI)
   - Docker containerization
   - Model serving

3. **Monitoring**
   - Performance tracking
   - Data drift detection
   - Automated retraining

## Dependencies

See `requirements.txt` for complete list:
- pandas
- numpy
- scikit-learn
- mlflow
- pytest (testing)
- Flask/FastAPI (API)

## Support

For issues or questions, check:
1. `pipeline.log` for detailed error messages
2. MLflow UI for experiment tracking
3. Test suite: `pytest tests/`

---

**Last Updated:** November 2025
