#imports necessary libraries
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
#imports the processed data files
from marketing_analysis.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

# This function prepares the features for the model training
def prepare_features(df: pd.DataFrame):
    X = df[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", "PreviousPurchases"]]
    X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)
    y = df["Conversion"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# This function trains the Logistic Regression model using GridSearchCV
def train_logistic_regression(X_train, y_train):
    logger.info("Training Logistic Regression with GridSearchCV...")
    param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid.fit(X_train, y_train)
    logger.success(f"Best Logistic Regression params: {grid.best_params_}")
    return grid.best_estimator_

# This function trains the Random Forest model using GridSearchCV
def train_random_forest(X_train, y_train):
    logger.info("Training Random Forest with GridSearchCV...")
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid.fit(X_train, y_train)
    logger.success(f"Best Random Forest params: {grid.best_params_}")
    return grid.best_estimator_

# This function trains the XGBoost model using GridSearchCV
def train_xgboost(X_train, y_train):
    logger.info("Training XGBoost with GridSearchCV...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1]
    }
    grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42), param_grid, cv=5)
    grid.fit(X_train, y_train)
    logger.success(f"Best XGBoost params: {grid.best_params_}")
    return grid.best_estimator_

# This function evaluates the model using classification report and ROC AUC score
def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    logger.info(f"Evaluation Report for {name}:")
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logger.info(f"ROC AUC Score: {roc_auc:.3f}")
    return y_pred

# This function runs the main script to train and evaluate the models
@app.command()
def main(features_path: Path = PROCESSED_DATA_DIR / "features.csv"):
    df = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test = prepare_features(df)

    models = {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train)
    }

    for name, model in models.items():
        evaluate_model(model, X_test, y_test, name)
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        logger.success(f"Saved {name} to {MODELS_DIR / f'{name}.pkl'}")

if __name__ == "__main__":
    app()
