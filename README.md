# UCI Heart Disease Classification

Binary classification pipeline predicting the presence of heart disease using the UCI Cleveland dataset.

## Setup

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

## Repository Structure

- `1_cleaning.ipynb` - Reads raw Cleveland dataset, cleans missing values, binarizes target variable, exports `heart_clean.csv`.
- `2_preprocessing.ipynb` - Data preprocessing, feature scaling, model comparison (Logistic Regression, Random Forest, XGBoost), hyperparameter tuning, evaluation, and SHAP explainability.
- `heart_clean.csv` - Processed output from cleaning step used as input for model training.
- `heart+disease/` - Raw dataset folder from UCI Machine Learning Repository.

## Workflow

1. Run `1_cleaning.ipynb` to generate `heart_clean.csv`.
2. Run `2_preprocessing.ipynb` to train models and evaluate metrics.

## Models & Evaluation

- Baseline models: Logistic Regression, Random Forest Classifier, XGBoost Classifier.
- Preprocessing: `StandardScaler` for age, `RobustScaler` for continuous features with outliers (`trestbps`, `chol`, `thalach`, `oldpeak`).
- Interpretability: SHAP summary and waterfall plots for feature importance analysis.
