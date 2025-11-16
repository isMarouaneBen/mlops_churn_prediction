"""
Data Ingestion Step
Loads raw data from Excel file and saves it as CSV
"""
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data(input_file: str, output_dir: str) -> str:
    """
    Load data from Excel and save as CSV
    
    Args:
        input_file: Path to input Excel file
        output_dir: Directory to save processed data
    
    Returns:
        Path to saved CSV file
    """
    logger.info(f"Loading data from {input_file}")
    
    # Load data from Excel
    df = pd.read_excel(input_file, sheet_name='E Comm')
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_path = Path(output_dir) / "raw_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    input_file = "../data/E Commerce Dataset.xlsx"
    output_dir = "../data/raw/training/raw"
    
    ingest_data(input_file, output_dir)
