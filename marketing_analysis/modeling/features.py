from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from marketing_analysis.config import PROCESSED_DATA_DIR


app = typer.Typer()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["AdvertisingPlatform", "AdvertisingTool"], errors="ignore")

def create_age_category(df: pd.DataFrame) -> pd.DataFrame:
    df["Age Category"] = pd.cut(
        df["Age"],
        bins=[13, 23, 33, 43, 53, 63, 73],
        labels=["13-23", "23-33", "33-43", "43-53", "53-63", "63-73"],
        right=False
    )
    return df

def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["WebsiteVisits", "PagesPerVisit", "TimeOnSite", "EmailOpens", "EmailClicks", "SocialShares"]])
    df["EngagementScore"] = scaled.sum(axis=1)
    df["EmailEngagementRatio"] = df["EmailClicks"] / df["EmailOpens"].replace(0, np.nan)
    return df

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "cleaned_marketing_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    logger.info("Generating features from dataset...")

    df = pd.read_csv(input_path)

    # Apply real features
    df = clean_data(df)
    df = create_age_category(df)
    df = create_engagement_features(df)

    df.to_csv(output_path, index=False)

    logger.success("Features generation complete. Saved to: {}".format(output_path))

if __name__ == "__main__":
    app()
