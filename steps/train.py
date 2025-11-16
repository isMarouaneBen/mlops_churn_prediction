from pathlib import Path
import pickle
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import yaml
import logging

logger = logging.getLogger(__name__)


def load_model_configs(config_path: str = "config.yaml") -> Dict[str, Dict[str, Any]]:
    """
    Load model configurations from YAML file
    
    Args:
        config_path: Path to the config.yaml file
        
    Returns:
        Dictionary with model configurations
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Map model class names to actual classes
    model_classes = {
        'LogisticRegression': LogisticRegression,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier
    }
    
    # Build model configs with actual class objects
    model_configs = {}
    for model_key, model_config in config.get('models', {}).items():
        class_name = model_config.get('model_class')
        if class_name in model_classes:
            model_configs[model_key] = {
                'model_class': model_classes[class_name],
                'params': model_config.get('params', {}),
                'display_name': model_config.get('display_name', class_name)
            }
    
    return model_configs


def evaluate_model(model, X_test, y_test, model_name: str) -> None:
    """
    Evaluate a trained model and log metrics
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Evaluating {model_name}")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")


def save_model(model, model_name: str, output_dir: Path) -> None:
    """
    Save a trained model to disk
    
    Args:
        model: Trained model object
        model_name: Name for the saved file
        output_dir: Directory to save the model
    """
    try:
        model_path = output_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved {model_name} to {model_path}")
    except Exception as e:
        logger.error(f"Error saving {model_name}: {e}")
        raise


def train_models(
    X_train_dict, 
    X_test_dict, 
    y_train, 
    y_test, 
    output_dir: str,
    dv=None,
    config_path: str = "config.yaml"
) -> None:
    """
    Train multiple classification models, evaluate them, and save to disk
    
    Args:
        X_train_dict: Training features (vectorized)
        X_test_dict: Test features (vectorized)
        y_train: Training target
        y_test: Test target
        output_dir: Directory to save models
        dv: DictVectorizer object to save (optional)
        config_path: Path to config.yaml file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model configurations from config.yaml
    model_configs = load_model_configs(config_path)
    
    # Train and evaluate each model
    for model_key, config in model_configs.items():
        try:
            logger.info(f"\nTraining {config['display_name']}...")
            
            # Initialize and train model
            model = config['model_class'](**config['params'])
            model.fit(X_train_dict, y_train)
            
            # Evaluate model
            evaluate_model(model, X_test_dict, y_test, config['display_name'])
            
            # Save model
            save_model(model, model_key, output_path)
            
        except Exception as e:
            logger.error(f"Error training {config['display_name']}: {e}")
            continue
    
    # Save DictVectorizer if provided
    if dv is not None:
        try:
            dv_path = output_path / "dict_vectorizer.pkl"
            with open(dv_path, 'wb') as f:
                pickle.dump(dv, f)
            logger.info(f"Saved DictVectorizer to {dv_path}")
        except Exception as e:
            logger.error(f"Error saving DictVectorizer: {e}")



if __name__ == "__main__":
    pass

