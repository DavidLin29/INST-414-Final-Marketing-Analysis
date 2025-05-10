#Import necessary libraries
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
#This imports data from your data folder
from marketing_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

#Initialize the app
app = typer.Typer()
#This function loads the raw file and puts it into a dataframe then returns it
def load_raw_data(filepath: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    return df
#Saves the processed dataframe into a csv format
def save_data(df: pd.DataFrame, output_path: Path):
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)

@app.command()
#This is the main function that runs the app and processes the data
def main(
    input_path: Path = RAW_DATA_DIR / "digital_marketing_campaign_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "cleaned_marketing_data.csv",
):
    logger.info("Processing dataset...")
    
    # Load
    df = load_raw_data(input_path)

    logger.info("Dataset preview:")
    logger.info(df.head())

    # Save
    save_data(df, output_path)
    logger.success("Processing complete.")

if __name__ == "__main__":
    app()
