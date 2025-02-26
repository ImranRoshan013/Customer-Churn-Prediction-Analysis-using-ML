Telecom Customer Churn Prediction
Project Overview
This project focuses on predicting customer churn for a telecom company using machine learning. The goal is to identify customers who are likely to leave the service so that the company can take proactive measures to retain them.

Key Steps
Data Cleaning & Preprocessing: Handled missing values, converted categorical variables, and scaled numerical features.

Exploratory Data Analysis (EDA): Analyzed the distribution of churn and relationships between features.

Feature Engineering: Created new features like TotalServices and AverageMonthlySpend.

Model Building: Trained and evaluated Logistic Regression, Random Forest, XGBoost, and an Ensemble Model.

Hyperparameter Tuning: Used RandomizedSearchCV to optimize model performance.

Model Evaluation: Achieved 84.3% accuracy with the Ensemble Model.

Model Deployment: Saved the best model for future use.

Tools & Technologies
Python

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn, XGBoost, SMOTE

Joblib (for model saving)

Results
Best Model: Ensemble Model (VotingClassifier)

Accuracy: 84.3%

Precision: 83.96%

Recall: 85.32%

F1-Score: 84.63%

ROC-AUC: 84.29%
