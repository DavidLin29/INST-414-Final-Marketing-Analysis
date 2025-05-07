# Marketing Conversion Prediction – INST414 Final Project

This project analyzes marketing campaign data to build predictive models that estimate whether a customer will convert based on engagement, demographics, and campaign features.

## 🔍 Project Overview

We used structured marketing campaign data to train and evaluate multiple predictive models. Our goal was to optimize marketing strategies by identifying the features that drive customer conversions.

This project was developed as part of the final assignment for INST414 – Data Science Methods at the University of Maryland.

## 🧠 Problem Statement

Can we predict whether a user will convert based on campaign and customer data?

Marketing teams often spend large sums targeting broad audiences. Predicting likely converters can significantly reduce costs and increase efficiency.

## 🛠️ Project Structure

This project follows the Cookiecutter Data Science structure:

├── data/
│   ├── raw/             <- Original data
│   ├── interim/         <- Cleaned, intermediate datasets
│   ├── processed/       <- Final datasets for modeling
│
├── models/              <- Trained model .pkl files
├── reports/
│   └── figures/         <- Visualizations like ROC curves and confusion matrices
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Marketing Analysis and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── Marketing Analysis   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Marketing Analysis a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

#How to run all the code
# 1. Run dataset cleaning and preparation
python -m marketing_analysis.modeling.dataset

# 2. Generate features
python -m marketing_analysis.modeling.features

# 3. Train all models (Logistic Regression, Random Forest, XGBoost)
python -m marketing_analysis.modeling.train

# 4. Run predictions on test set
python -m marketing_analysis.modeling.predict

# 5. Visualize model performance and do error analysis
python -m marketing_analysis.modeling.visualize_and_error_analysis plot-confusion-matrix
python -m marketing_analysis.modeling.visualize_and_error_analysis plot-roc-curves
python -m marketing_analysis.modeling.visualize_and_error_analysis plot-feature-importance
python -m marketing_analysis.modeling.visualize_and_error_analysis find-errors
├── notebooks/           <- Jupyter notebooks for exploration
├── marketing_analysis/  <- All source code
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   └── modeling/
│       ├── train.py
│       ├── predict.py
│       └── visualize_and_error_analysis.py

## ⚙️ Setup and Usage

### 📦 Environment Setup

Use Conda to create the environment:

conda env create -f environment.yml
conda activate marketing_analysis

### ▶️ Run the Full Pipeline

python -m marketing_analysis.dataset
python -m marketing_analysis.features
python -m marketing_analysis.modeling.train
python -m marketing_analysis.modeling.predict
python -m marketing_analysis.modeling.visualize_and_error_analysis plot-confusion-matrix

Other available visualizations:

python -m marketing_analysis.modeling.visualize_and_error_analysis plot-roc-curves
python -m marketing_analysis.modeling.visualize_and_error_analysis plot-feature-importance
python -m marketing_analysis.modeling.visualize_and_error_analysis find-errors

## 🤖 Models Trained

We implemented and compared three models:

Model               | Accuracy | F1 Score | ROC AUC
--------------------|----------|----------|---------
Logistic Regression | 88%      | 84%      | 0.71
Random Forest       | 88%      | 84%      | 0.71
XGBoost             | 89%      | 85%      | 0.72

All models were tuned using GridSearchCV.

## 🧠 Key Features Used

- Engagement Score
- Income
- Ad Spend
- Age Category (encoded)
- Campaign Channel (encoded)
- Previous Purchases

## 📊 Visual Output

- Confusion Matrix
- ROC Curve
- Feature Importance
- Error Samples (misclassified predictions)

These are saved in the reports/figures/ directory.

## 💡 Business Value

By accurately predicting conversion likelihood:
- Marketing campaigns can focus on high-probability customers
- Budgets can be reduced by avoiding outreach to low-conversion segments
- Messaging can be tailored to audience segments with the highest impact


## 👨‍💻 Author

David Lin  
University of Maryland  
Spring 2025 – INST414 Final Project

## 📜 License

MIT License


#Author Note
This was completed with the help of an AI assistance tool(Github Copilot, A little ChatGPT) Also confusion matrix is the most helpful visualization in my opinion