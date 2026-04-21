import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def fetch_data(file_path: str | Path) -> pd.DataFrame:
    """
    Fetch data from the raw CSV file.
    
    Args:
        file_path (str | Path): Path to the raw data file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    logger.info(f"Fetching data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        logger.error(f"Error reading data from {file_path}: {e}")
        raise
