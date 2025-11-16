"""
Simple unit tests for data cleaning
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from steps.clean import clean_data


@pytest.fixture
def sample_raw_data():
    """Create sample raw data with issues"""
    with tempfile.TemporaryDirectory() as temp_dir:
        data = {
            'Customer ID': [1, 2, 3, 4, 5],
            'Gender': ['Male', 'Female', None, 'Male', 'Female'],
            'Purchase Amount': [100.5, 200.0, np.nan, 150.0, 300.0],
            'Churn': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, "raw_data.csv")
        df.to_csv(file_path, index=False)
        yield file_path, temp_dir


def test_missing_values_filled(sample_raw_data):
    """Test that missing values are filled with 0"""
    input_file, output_dir = sample_raw_data
    
    # Run cleaning
    output_path = clean_data(input_file, output_dir)
    
    # Load cleaned data
    cleaned_df = pd.read_csv(output_path)
    
    # Assert no missing values
    assert cleaned_df.isna().sum().sum() == 0


def test_column_names_standardized(sample_raw_data):
    """Test that column names are lowercase with underscores"""
    input_file, output_dir = sample_raw_data
    
    output_path = clean_data(input_file, output_dir)
    cleaned_df = pd.read_csv(output_path)
    
    # Check column names
    assert 'customer_id' in cleaned_df.columns
    assert 'gender' in cleaned_df.columns
    assert all(col.islower() for col in cleaned_df.columns)


def test_string_values_standardized(sample_raw_data):
    """Test that string values are lowercase"""
    input_file, output_dir = sample_raw_data
    
    output_path = clean_data(input_file, output_dir)
    cleaned_df = pd.read_csv(output_path)
    
    # Check string values
    assert all(val.islower() for val in cleaned_df['gender'].unique() if isinstance(val, str))


def test_data_shape_preserved(sample_raw_data):
    """Test that row and column counts are preserved"""
    input_file, output_dir = sample_raw_data
    
    original_df = pd.read_csv(input_file)
    output_path = clean_data(input_file, output_dir)
    cleaned_df = pd.read_csv(output_path)
    
    # Check dimensions
    assert len(cleaned_df) == len(original_df)
    assert len(cleaned_df.columns) == len(original_df.columns)


def test_output_file_created(sample_raw_data):
    """Test that cleaned data file is created"""
    input_file, output_dir = sample_raw_data
    
    output_path = clean_data(input_file, output_dir)
    
    # Check file exists
    assert os.path.exists(output_path)
    assert output_path.endswith("cleaned_data.csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
