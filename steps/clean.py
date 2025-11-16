"""
Data Cleaning and Preprocessing Step
Handles missing values, standardizes column names, and prepares data for modeling
"""
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(input_file: str, output_dir: str) -> str:
    """
    Clean and preprocess data
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save cleaned data
    
    Returns:
        Path to saved cleaned data CSV
    """
    logger.info(f"Loading raw data from {input_file}")
    df = pd.read_csv(input_file)
    
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Missing values:\n{df.isna().sum()}")
    
    # Handle missing values
    df.fillna(0, inplace=True)
    logger.info("Filled missing values with 0")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    logger.info("Standardized column names to lowercase with underscores")
    
    # Standardize string values
    df = df.apply(lambda x: x.str.lower().str.replace(' ', '_') if x.dtype == "object" else x)
    logger.info("Standardized string values to lowercase")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    output_path = Path(output_dir) / "cleaned_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")
    logger.info(f"Final shape: {df.shape}")
    
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    input_file = "../data/raw/raw_data.csv"
    output_dir = "../data/training/processed"
    
    clean_data(input_file, output_dir)
