#imports necessary libraries
from pathlib import Path
import typer
import pandas as pd
import joblib
from loguru import logger
from tqdm import tqdm
#imports the processed data files
from marketing_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
#This function runs the models and makes predictions on the data
@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    predictions_dir: Path = PROCESSED_DATA_DIR
):
    logger.info("Loading feature data...")
    df = pd.read_csv(features_path)

    logger.info("Preparing features for prediction...")
    X = df[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", "PreviousPurchases"]]
    X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)

    model_names = ["logistic_regression", "random_forest", "xgboost"]

    for name in model_names:
        model_path = MODELS_DIR / f"{name}.pkl"
        logger.info(f"Loading {name} model from {model_path}...")
        model = joblib.load(model_path)

        logger.info(f"Making predictions with {name}...")
        predictions = model.predict(X)
        df[f"Predicted_{name}"] = predictions

        output_path = predictions_dir / f"predictions_{name}.csv"
        df.to_csv(output_path, index=False)
        logger.success(f"Saved predictions to {output_path}")

    logger.success("All model predictions completed.")

if __name__ == "__main__":
    app()
