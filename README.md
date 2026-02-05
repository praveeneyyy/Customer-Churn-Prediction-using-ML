# Customer-Churn-Prediction-using-ML
About This Application
Overview
This Customer Churn Prediction System uses Machine Learning to predict whether a customer is likely to leave a telecommunications service provider.

Dataset
Source: Telco Customer Churn Dataset
Total Records: ~7,043 customers
Target Variable: Churn (Yes/No)
Model Development
The model was developed using:

Data Preprocessing: Label encoding for categorical variables
Handling Imbalance: SMOTE (Synthetic Minority Oversampling Technique)
Model Selection: Random Forest Classifier
Hyperparameter Tuning: GridSearchCV with 5-fold cross-validation
Key Features Influencing Churn
Contract type (Month-to-month customers churn more)
Tenure (Longer tenure = lower churn)
Internet service type
Tech support availability
Online security services
Use Cases
Customer Retention: Identify at-risk customers for proactive retention
Resource Allocation: Focus retention efforts on high-risk segments
Business Planning: Understand churn patterns and trends
Important Notes
⚠️ This model should be used as a tool to support decision-making, not as the sole basis for critical business decisions.

Built with: Streamlit, Scikit-learn, XGBoost
