
# Loan Approval Prediction

## Overview
This project aims to predict loan approval outcomes using machine learning models trained on financial and personal attributes of loan applicants. The pipeline includes comprehensive data preprocessing, exploratory data analysis (EDA), and the use of optimized models like Decision Tree and XGBoost for accurate predictions.

---

## Features
- **Data Preprocessing**: Cleaning data, encoding categorical variables, handling missing values, and feature engineering by aggregating asset-related columns.
- **Exploratory Data Analysis (EDA)**:
  - Visualization of loan status distribution.
  - Histograms of numerical features for distribution analysis.
  - Correlation matrix heatmap to analyze relationships between features.
  - Box plots to understand relationships between features like income and loan approval status.
- **Machine Learning Models**:
  - Decision Tree Classifier (optimized using GridSearchCV).
  - XGBoost Classifier (optimized using RandomizedSearchCV).
  - Navie Bayes Classifier
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, and ROC curve visualization.

---

## Dataset
The dataset contains the following features:
- **Income Details**: `income_annum`, `loan_amount`, `loan_term`.
- **Creditworthiness**: `cibil_score`.
- **Assets**: Aggregated total asset value.
- **Categorical Variables**:
  - `education` (Graduate/Not Graduate).
  - `self_employed` (Yes/No).
  - `loan_status` (Approved/Rejected) - Target variable.

### Preprocessing Steps
1. Dropped irrelevant columns (e.g., `loan_id`).
2. Aggregated asset-related columns into a single `Assets` feature.
3. Cleaned and encoded categorical variables into numerical format.
4. Scaled numerical features using `StandardScaler`.

---

## Prerequisites
- **Python 3.x**
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `seaborn`
  - `matplotlib`
  - `xgboost`

Usage:
`use final_code_version3.ipynb file`

Install the required libraries:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost

