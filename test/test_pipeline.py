import pytest
import pandas as pd
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processor import clean_data

@pytest.fixture
def mock_raw_data():
    """
    Creates a small mock DataFrame mimicking the raw dataset's structure.
    """
    return pd.DataFrame({
        'country': ['India', 'China', 'United States', None],
        'continent': ['Asia', 'Asia', 'North America', 'Unknown'],
        '2023 population': [1428627663, 1425671352, 339996563, 1000],
        'growth rate': ['0.81%', '-0.02%', '0.50%', '0.00%'],
        'world percentage': ['17.85%', '17.81%', '4.25%', '0.01%'],
        'density (km²)': [481, 151, 37, 10]
    })

def test_clean_data_column_names(mock_raw_data):
    """
    Tests if the column names are properly snake_cased and standardized.
    """
    cleaned_df = clean_data(mock_raw_data)
    
    # Check if spaces were replaced with underscores
    assert '2023_population' in cleaned_df.columns
    assert 'growth_rate' in cleaned_df.columns
    assert 'world_percentage' in cleaned_df.columns
    assert 'density_(km²)' in cleaned_df.columns

def test_clean_data_percentage_conversion(mock_raw_data):
    """
    Tests if string percentages are properly converted to floats.
    """
    cleaned_df = clean_data(mock_raw_data)
    
    # Check if conversion worked (0.81% -> 0.0081)
    assert cleaned_df.loc[cleaned_df['country'] == 'India', 'growth_rate'].values[0] == pytest.approx(0.0081)
    assert cleaned_df.loc[cleaned_df['country'] == 'China', 'growth_rate'].values[0] == pytest.approx(-0.0002)

def test_clean_data_null_dropping(mock_raw_data):
    """
    Tests if rows with missing critical data (like country name) are dropped.
    """
    # mock_raw_data has 4 rows, 1 has a None for country
    assert len(mock_raw_data) == 4
    
    cleaned_df = clean_data(mock_raw_data)
    
    # The row with None country should be dropped
    assert len(cleaned_df) == 3
    assert cleaned_df['country'].isnull().sum() == 0
