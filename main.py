import os
import sys
import logging
import yaml
import pickle
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Import step functions
from steps.ingest import ingest_data
from steps.clean import clean_data
from steps.train import train_models, load_model_configs, evaluate_model
from steps.predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(data_file: str, config_path: str = "config.yaml"):
    """
    Run the complete ML pipeline: ingest -> clean -> train
    
    Args:
        data_file: Path to input Excel file
        config_path: Path to config.yaml
    """
    logger.info("=" * 60)
    logger.info("STARTING ML PIPELINE")
    logger.info("=" * 60)
    
    # Setup directories
    raw_dir = "data/training/raw"
    processed_dir = "data/training/processed"
    models_dir = "models"
    
    # Step 1: Ingest data
    logger.info("\n[STEP 1] Ingesting data...")
    raw_data_path = ingest_data(data_file, raw_dir)
    
    # Step 2: Clean data
    logger.info("\n[STEP 2] Cleaning data...")
    cleaned_data_path = clean_data(raw_data_path, processed_dir)
    
    # Step 3: Train models
    logger.info("\n[STEP 3] Training models...")
    train_and_evaluate(cleaned_data_path, models_dir, config_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


def train_and_evaluate(data_path: str, models_dir: str, config_path: str):
    """
    Train models and save them
    
    Args:
        data_path: Path to cleaned CSV file
        models_dir: Directory to save models
        config_path: Path to config.yaml
    """
    import pandas as pd
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    if 'churn' not in df.columns:
        logger.warning("'churn' column not found. Using all columns as features.")
        X = df
        y = None
    else:
        y = df['churn']
        X = df.drop('churn', axis=1)
    
    # Vectorize features
    dv = DictVectorizer(sparse=False)
    X_dict = X.to_dict(orient='records')
    X_vectorized = dv.fit_transform(X_dict)
    
    # Train-test split
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test = train_test_split(X_vectorized, test_size=0.2, random_state=42)
        y_train, y_test = None, None
    
    # Train models
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    train_models(X_train, X_test, y_train, y_test, models_dir, dv=dv, config_path=config_path)


def train_with_mlflow(data_file: str, config_path: str = "config.yaml", experiment_name: str = "churn_prediction"):
    """
    Run pipeline with MLflow tracking: logs metrics, parameters, and models
    
    Args:
        data_file: Path to input Excel file
        config_path: Path to config.yaml
        experiment_name: MLflow experiment name
    """
    import pandas as pd
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import train_test_split
    
    logger.info("=" * 60)
    logger.info("STARTING ML PIPELINE WITH MLFLOW")
    logger.info("=" * 60)
    
    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="full_pipeline"):
        # Setup directories
        raw_dir = "data/training/raw"
        processed_dir = "data/training/processed"
        models_dir = "models"
        
        try:
            # Step 1: Ingest
            logger.info("\n[STEP 1] Ingesting data...")
            raw_data_path = ingest_data(data_file, raw_dir)
            
            # Step 2: Clean
            logger.info("\n[STEP 2] Cleaning data...")
            cleaned_data_path = clean_data(raw_data_path, processed_dir)
            
            # Step 3: Train with MLflow
            logger.info("\n[STEP 3] Training models with MLflow...")
            
            # Load data
            df = pd.read_csv(cleaned_data_path)
            
            if 'churn' not in df.columns:
                logger.error("'churn' column not found in data")
                raise ValueError("Missing 'churn' target column")
            
            y = df['churn']
            X = df.drop('churn', axis=1)
            
            # Vectorize
            dv = DictVectorizer(sparse=False)
            X_dict = X.to_dict(orient='records')
            X_vectorized = dv.fit_transform(X_dict)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, test_size=0.2, random_state=42
            )
            
            # Log parameters
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("n_features", X_vectorized.shape[1])
            
            # Load model configs
            model_configs = load_model_configs(config_path)
            
            # Train each model
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            
            for model_key, config in model_configs.items():
                try:
                    logger.info(f"\nTraining {config['display_name']}...")
                    
                    # Initialize and train
                    model = config['model_class'](**config['params'])
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    auc = roc_auc_score(y_test, y_pred_proba)
                    
                    # Log metrics with model prefix
                    mlflow.log_metric(f"{model_key}_accuracy", accuracy)
                    mlflow.log_metric(f"{model_key}_precision", precision)
                    mlflow.log_metric(f"{model_key}_recall", recall)
                    mlflow.log_metric(f"{model_key}_f1", f1)
                    mlflow.log_metric(f"{model_key}_auc", auc)
                    
                    logger.info(f"✓ {config['display_name']} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
                    
                    # Save model
                    model_path = Path(models_dir) / f"{model_key}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(str(model_path))
                    
                except Exception as e:
                    logger.error(f"Error training {config['display_name']}: {e}")
                    continue
            
            # Save and log vectorizer
            dv_path = Path(models_dir) / "dict_vectorizer.pkl"
            with open(dv_path, 'wb') as f:
                pickle.dump(dv, f)
            mlflow.log_artifact(str(dv_path))
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE WITH MLFLOW COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            mlflow.log_param("status", "failed")
            raise


if __name__ == "__main__":
    # Run simple pipeline
    # run_pipeline("data/E Commerce Dataset.xlsx")
    
    # Or run with MLflow
    train_with_mlflow("data/E Commerce Dataset.xlsx")
