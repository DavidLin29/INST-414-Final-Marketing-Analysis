from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer

from marketing_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


app = typer.Typer()

def load_raw_data(filepath: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    return df

def save_data(df: pd.DataFrame, output_path: Path):
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "digital_marketing_campaign_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "cleaned_marketing_data.csv",
):
    logger.info("Processing dataset...")
    
    # Load
    df = load_raw_data(input_path)

    # Example transformation (replace with real feature functions if needed)
    logger.info("Dataset preview:")
    logger.info(df.head())

    # Save
    save_data(df, output_path)
    logger.success("Processing complete.")

if __name__ == "__main__":
    app()
