Customer Churn Prediction
This project focuses on predicting customer churn for a telecom company using machine learning. The goal is to identify customers who are likely to churn (leave) the service, enabling the company to take proactive measures to retain them.

Table of Contents
Problem Statement

Dataset

Approach

Results

Technologies Used

How to Use

Future Work

Problem Statement
Customer churn is a critical issue for businesses, as retaining existing customers is often more cost-effective than acquiring new ones. This project aims to predict customer churn using historical data and provide actionable insights to reduce churn rates.

Dataset
The dataset used is the Telco Customer Churn Dataset from IBM. It contains information about:

Customer demographics (e.g., gender, age, partner, dependents).

Services signed up for (e.g., phone, internet, online security).

Account information (e.g., tenure, contract type, payment method).

Target variable: Churn (Yes/No).

Approach
Data Preprocessing:

Handled missing values and encoded categorical variables.

Performed feature engineering (e.g., created tenure groups, monthly-to-total charges ratio).

Exploratory Data Analysis (EDA):

Visualized key trends and correlations.

Analyzed the distribution of churned vs. non-churned customers.

Model Building:

Trained and evaluated multiple models, including Balanced Random Forest, EasyEnsemble, and XGBoost.

Selected XGBoost as the best model due to its high accuracy (78%) and balanced precision-recall trade-off.

Results
Best Model: XGBoost achieved an accuracy of 78%, a precision of 56%, and a recall of 70% for churn prediction.

Key Insights:

Customers with longer tenure and two-year contracts are less likely to churn.

Customers with fiber optic internet and higher monthly charges are more likely to churn.

Technologies Used
Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn.

Tools: Jupyter Notebook, Git.
