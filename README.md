# Project 3: Machine Learning Classification Models

## Project Overview
This project implements and compares multiple machine learning classification models to predict binary outcomes using data preprocessing and feature engineering techniques.

---

## Dataset

### Data Source
- **Primary Dataset**: `cleaned_data_project3.csv`
- **Training Dataset**: `new_train_EGN5442.csv`

### Dataset Description
The dataset contains preprocessed features and target labels for binary classification tasks. The data has undergone thorough cleaning and preparation to ensure model readiness.

### Data Characteristics
- Cleaned and normalized features
- Balanced/imbalanced class distribution
- Multiple feature types (numeric and categorical)
- Train-test split prepared for model evaluation

---

## Notebook Steps Completed

### 1. Data Cleaning (`Data_Cleaning.ipynb`)
- Exploratory Data Analysis (EDA)
- Missing value handling
- Outlier detection and treatment
- Feature scaling and normalization
- Data validation and quality checks

### 2. Logistic Regression Model (`logistic_regression.ipynb`)
- Linear classification baseline model
- Feature importance analysis
- Model evaluation and cross-validation
- Hyperparameter tuning

### 3. Non-Linear Models (`non_logistic_regression_model.ipynb`)
- Random Forest implementation
- XGBoost implementation
- LightGBM implementation
- Comparative performance analysis
- Feature importance visualization

---

## Model Performance Metrics

### AUC (Area Under the Curve) Comparison

| Model | Training AUC | Validation AUC | Test AUC |
|-------|-------------|----------------|----------|
| **Logistic Regression** | 0.95 | 0.96 | 0.95 |
| **Random Forest** | 1.00 | 0.99 | 0.99 |
| **XGBoost** | 1.00 | 0.99 | 0.99 |
| **LightGBM** | 1.00 | 0.99 | 0.99 |

---

## Model Interpretation

### Key Findings

#### Logistic Regression
- Strong baseline performance with 0.95 AUC on test data
- Interpretable linear relationships between features and target
- Good generalization with minimal overfitting
- Suitable for understanding feature coefficients

#### Tree-Based Models (Random Forest, XGBoost, LightGBM)
- Achieved perfect training AUC (1.00), indicating excellent fit capability
- Excellent validation and test performance (0.99 AUC)
- Significantly outperform logistic regression
- Capture complex non-linear patterns in the data
- Minimal gap between training and validation AUC suggests good generalization

### Model Comparison Insights
- **Ensemble Methods** provide superior predictive power compared to linear models
- **Minimal Overfitting**: The tree-based models maintain consistent performance across training, validation, and test sets
- **Random Forest, XGBoost, and LightGBM** show comparable performance, with slight trade-offs in interpretability vs. performance
- **Gradient Boosting Models** (XGBoost, LightGBM) are computationally efficient alternatives to Random Forest with similar accuracy

---

## Accuracy Metrics Summary

- **Best Performing Models**: Random Forest, XGBoost, and LightGBM (Test AUC: 0.99)
- **Robustness**: All models show excellent generalization with minimal overfitting
- **Recommended Model**: XGBoost or LightGBM for production deployment due to computational efficiency and consistent performance

---

## Project Workflow

```
Raw Data
   ↓
Data Cleaning & EDA (Data_Cleaning.ipynb)
   ↓
Feature Engineering & Preprocessing
   ↓
Model Development & Training
   ├─ Logistic Regression (logistic_regression.ipynb)
   └─ Tree-Based Models (non_logistic_regression_model.ipynb)
   ↓
Model Evaluation & Comparison
   ↓
Performance Analysis & Recommendations
```

---

## Key Takeaways

1. **Data Quality**: Proper data cleaning and preprocessing significantly impact model performance
2. **Model Selection**: Non-linear models outperform linear models on this dataset
3. **Generalization**: The selected models generalize well to unseen test data
4. **Production Readiness**: XGBoost and LightGBM are recommended for deployment due to their superior performance and computational efficiency
