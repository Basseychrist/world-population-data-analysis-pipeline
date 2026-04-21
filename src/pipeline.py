import logging
import os
from pathlib import Path
from data_fetcher import fetch_data
from data_processor import clean_data, save_data

def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log", mode='w')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")
    return logger

def run_pipeline():
    """
    Run the end-to-end data pipeline.
    """
    logger = setup_logging()
    
    # Paths setup
    BASE_DIR = Path(__file__).parent.parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "world_population_data.csv"
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "world_population_data_clean.csv"
    
    logger.info("Starting Data Pipeline execution...")

    if not RAW_DATA_PATH.exists():
        logger.error(f"Cannot find raw data file: {RAW_DATA_PATH}")
        return
    
    try:
        # Step 1: Fetch
        raw_df = fetch_data(RAW_DATA_PATH)
        
        # Step 2: Process
        clean_df = clean_data(raw_df)
        
        # Step 3: Save
        save_data(clean_df, PROCESSED_DATA_PATH)
        
        logger.info("Data Pipeline executed successfully.")
        
    except Exception as e:
        logger.error(f"Data Pipeline execution failed: {e}")
        
if __name__ == "__main__":
    run_pipeline()
