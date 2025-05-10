#Import necessary libraries
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Imports the processed data files
from marketing_analysis.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
#Creates a heatmap of the correlation matrix of the dataframe
def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(20, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved heatmap to {save_path}")
#When you run the script from the command line it will run this function
@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "features.csv",
    output_path: Path = FIGURES_DIR / "correlation_heatmap.png",
):
    logger.info("Generating plot from data...")

    df = pd.read_csv(input_path)

    plot_correlation_heatmap(df, output_path)

    logger.success("Plot generation complete.")

if __name__ == "__main__":
    app()
