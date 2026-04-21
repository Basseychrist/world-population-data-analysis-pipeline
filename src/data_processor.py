import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the population data.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    logger.info("Starting data cleaning process...")
    
    # Create a copy so we don't mutate the original in-place
    processed_df = df.copy()
    
    # 1. Standardize column names (lowercase, replace spaces with underscores)
    processed_df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True)
    
    # 2. Convert percentage columns from string to float
    for col in ['growth_rate', 'world_percentage']:
        if col in processed_df.columns:
            logger.info(f"Converting `{col}` from string to float...")
            processed_df[col] = processed_df[col].astype(str).str.rstrip('%').astype('float') / 100.0
            
    # 3. Handle missing values (basic handling: dropping rows with nulls in critical columns)
    initial_len = len(processed_df)
    processed_df.dropna(subset=['country', '2023_population'], inplace=True)
    if initial_len > len(processed_df):
        logger.warning(f"Dropped {initial_len - len(processed_df)} rows due to missing critical values.")
        
    logger.info("Data cleaning completed.")
    return processed_df

def save_data(df: pd.DataFrame, output_path: str | Path):
    """
    Saves the processed dataframe to the specified file path.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str | Path): Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving processed data to {output_path}...")
    try:
        df.to_csv(output_path, index=False)
        logger.info("Data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save data to {output_path}: {e}")
        raise
