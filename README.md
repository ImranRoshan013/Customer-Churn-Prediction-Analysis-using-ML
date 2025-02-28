# Telco Customer Churn Prediction

This project predicts customer churn for a telecommunications company using machine learning. It combines robust data preprocessing, feature engineering, model tuning, and ensembling to improve prediction accuracy. The final model is deployed for real-world use.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Ensemble Modeling](#ensemble-modeling)
8. [Results](#results)
9. [Model Deployment](#model-deployment)
10. [Contributing](#contributing)
11. [License](#license)

---

## Introduction
Customer churn is a critical metric for telecom companies. This project aims to predict whether a customer will leave the service (churn) based on their demographics, account information, and service usage. The model is trained using **Logistic Regression**, **Random Forest**, and **XGBoost**, and an **ensemble model** is created to improve performance.

---

## Dataset
The dataset used is the **Telco Customer Churn Dataset** from Kaggle. It contains 7,043 rows and 21 columns, including customer demographics, account information, and service usage.

### Dataset Summary
- **Total Rows**: 7,043
- **Columns**: 21
- **Target Variable**: `Churn` (Yes/No)
- **Key Features**:
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Services: `PhoneService`, `InternetService`, `OnlineSecurity`, etc.
  - Account Information: `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`

### Dataset Preview
```plaintext
   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService  \
0  7590-VHVEG  Female              0     Yes         No       1           No   
1  5575-GNVDE    Male              0      No         No      34          Yes   
2  3668-QPYBK    Male              0      No         No       2          Yes   
3  7795-CFOCW    Male              0      No         No      45           No   
4  9237-HQITU  Female              0      No         No       2          Yes   

      MultipleLines InternetService OnlineSecurity  ... DeviceProtection  \
0  No phone service             DSL             No  ...               No   
1                No             DSL            Yes  ...              Yes   
2                No             DSL            Yes  ...               No   
3  No phone service             DSL            Yes  ...              Yes   
4                No     Fiber optic             No  ...               No   

  TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling  \
0          No          No              No  Month-to-month              Yes   
1          No          No              No        One year               No   
2          No          No              No  Month-to-month              Yes   
3         Yes          No              No        One year               No   
4          No          No              No  Month-to-month              Yes   

               PaymentMethod MonthlyCharges  TotalCharges Churn  
0           Electronic check          29.85         29.85    No  
1               Mailed check          56.95        1889.5    No  
2               Mailed check          53.85        108.15   Yes  
3  Bank transfer (automatic)          42.30       1840.75    No  
4           Electronic check          70.70        151.65   Yes  
```

---

## Exploratory Data Analysis (EDA)
### Key Insights
1. **Churn Distribution**:
   - 26.5% of customers churned.
   - Visualized using a count plot with annotations.

2. **Numerical Feature Distributions**:
   - `tenure`, `MonthlyCharges`, `TotalCharges`, and `AverageMonthlySpend` were analyzed using histograms.

3. **Correlation Heatmap**:
   - Identified relationships between numerical features.

---

## Feature Engineering
### Key Transformations
1. **Binary Encoding**:
   - Converted `Yes/No` columns to `1/0`.

2. **New Features**:
   - `TotalServices`: Count of subscribed services.
   - `AverageMonthlySpend`: Normalized spending by tenure.

3. **One-Hot Encoding**:
   - Converted categorical variables into numeric features.

---

## Model Training and Evaluation
### Models Used
1. **Logistic Regression**:
   - Accuracy: 81.11%
   - ROC-AUC: 81.10%

2. **Random Forest**:
   - Accuracy: 84.06%
   - ROC-AUC: 84.06%

3. **XGBoost**:
   - Accuracy: 83.33%
   - ROC-AUC: 83.32%

### Evaluation Metrics
- **Accuracy**: Overall correctness.
- **Precision**: Correctness of positive predictions.
- **Recall**: Ability to capture positive cases.
- **F1-Score**: Balance between precision and recall.
- **ROC-AUC**: Model's discriminative ability.

---

## Hyperparameter Tuning
### Random Forest
- **Best Parameters**:
  ```plaintext
  {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': None}
  ```
- **Tuned Accuracy**: 84.15%

### XGBoost
- **Best Parameters**:
  ```plaintext
  {'subsample': 0.9, 'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.2, 'colsample_bytree': 1.0}
  ```
- **Tuned Accuracy**: 82.95%

---

## Ensemble Modeling
### Voting Classifier
- **Accuracy**: 84.30%
- **ROC-AUC**: 84.29%

### Cross-Validation
- **Cross-Validation Accuracy**: 83.57%

---

## Results
### What You Did
- **Data Preprocessing**: Cleaned the dataset by handling missing values, encoding categorical variables, and creating new features like `TotalServices` and `AverageMonthlySpend`.
- **Exploratory Data Analysis (EDA)**: Analyzed the dataset to understand the distribution of churn, numerical features, and correlations between variables.
- **Model Training**: Trained three baseline models: **Logistic Regression**, **Random Forest**, and **XGBoost**.
- **Hyperparameter Tuning**: Used **RandomizedSearchCV** to optimize hyperparameters for Random Forest and XGBoost.
- **Ensemble Modeling**: Combined the best-performing models into a **Voting Classifier** to improve prediction accuracy.
- **Model Evaluation**: Evaluated models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

### Why You Did It
- **Business Problem**: Customer churn is a critical issue for telecom companies. Predicting churn helps businesses take proactive measures to retain customers.
- **Model Performance**: By experimenting with multiple models and ensembling, we aimed to achieve the highest possible accuracy and robustness.
- **Explainability**: Feature engineering and EDA helped identify key drivers of churn, making the model interpretable for stakeholders.

### What Were the Results
- **Best Model**: The **Ensemble Model (Voting Classifier)** achieved the highest accuracy of **84.30%** and an ROC-AUC score of **84.29%**.
- **Cross-Validation**: The ensemble model demonstrated consistent performance with a cross-validation accuracy of **83.57%**.
- **Key Insights**:
  - Customers with higher `MonthlyCharges` and lower `tenure` are more likely to churn.
  - The `Contract` type (month-to-month vs. long-term) is a significant predictor of churn.
- **Deployment**: The final model was saved as `final_churn_ensemble_model.pkl` for deployment in production environments.

---

## Model Deployment
The final ensemble model is saved as `final_churn_ensemble_model.pkl` for deployment. It can be used in production environments to predict customer churn.

```python
import joblib

# Load the model
model = joblib.load("final_churn_ensemble_model.pkl")

# Predict on new data
predictions = model.predict(new_data)
```

---

## Contributing
Contributions to this project are welcome! If you have any improvements or suggestions, feel free to create a pull request or open an issue.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
