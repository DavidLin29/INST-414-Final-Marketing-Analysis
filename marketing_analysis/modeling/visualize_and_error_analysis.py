#imports necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
from marketing_analysis.config import PROCESSED_DATA_DIR, MODELS_DIR

import typer
app = typer.Typer()

@app.command()
def plot_confusion_matrix():
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    y_true = df["Conversion"]

    model_names = ["logistic_regression", "random_forest", "xgboost"]
    for name in model_names:
        model = joblib.load(MODELS_DIR / f"{name}.pkl")
        X = df[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", "PreviousPurchases"]]
        X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)
        y_pred = model.predict(X)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(PROCESSED_DATA_DIR / f"confusion_matrix_{name}.png")
        plt.close()

@app.command()
def plot_roc_curves():
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    y_true = df["Conversion"]
    X = df[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", "PreviousPurchases"]]
    X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)

    model_names = ["logistic_regression", "random_forest", "xgboost"]

    plt.figure(figsize=(8, 6))
    for name in model_names:
        model = joblib.load(MODELS_DIR / f"{name}.pkl")
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / "roc_curves_all_models.png")
    plt.close()

@app.command()
def plot_feature_importance():
    model = joblib.load(MODELS_DIR / "xgboost.pkl")
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    X = df[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", "PreviousPurchases"]]
    X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)
    importance = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / "feature_importance_xgboost.png")
    plt.close()

@app.command()
def find_errors(model_name: str = "xgboost"):
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    model = joblib.load(MODELS_DIR / f"{model_name}.pkl")

    X = df[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", "PreviousPurchases"]]
    X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)
    y_true = df["Conversion"]
    y_pred = model.predict(X)

    df["Predicted"] = y_pred
    errors = df[df["Conversion"] != df["Predicted"]]
    sample_errors = errors.sample(n=min(10, len(errors)))
    sample_errors.to_csv(PROCESSED_DATA_DIR / f"error_analysis_{model_name}.csv", index=False)
    print(sample_errors)

if __name__ == "__main__":
    app()
