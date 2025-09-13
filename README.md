1)Regression Project

2)Objective:
This project predicts daily bike rentals using the Bike Sharing Dataset (2011–2012). It demonstrates an end-to-end ML workflow: preprocessing → EDA → feature engineering → regression models → ensemble methods → ANN → model comparison.

3)dataset:
Source: UCI Bike Sharing Dataset
Features:
Temporal: datetime → year, month, day, hour, weekday, weekend
Weather: season, weather, temp, atemp, humidity, windspeed
Flags: holiday, workingday
Target: count (total rentals = casual + registered)

4)Methodology:
🔹 Exploratory Data Analysis (EDA)
Demand distribution plots
Time-based demand trends (daily/seasonal)
Correlation heatmaps
Weather and working-day impact on rentals
🔹 Preprocessing
Feature extraction from datetime
Log-transform applied on target variable (count)
🔹 Modeling
Linear Regression
KNN Regressor
Support Vector Regressor (SVR)
Decision Tree (default & tuned)
Random Forest (default & tuned)
Bagging Regressor (default & tuned)
AdaBoost (default & tuned)
Gradient Boosting (default & tuned)
XGBoost (default & tuned)
Artificial Neural Network (ANN)
🔹 Feature Selection & Multicollinearity
Applied Variance Inflation Factor (VIF) to reduce multicollinearity
Retrained selected models after feature reduction
🔹 Experimentation
Multiple train-test splits tested: 60:40, 65:35, 70:30, 75:25, 80:20
Compared model stability across splits (ANN trained only on one split)

5) Results:
✅ Ensemble methods (Random Forest, Gradient Boosting, XGBoost) consistently outperformed linear models
✅ ANN achieved strong accuracy on the chosen split
✅ Log-transform improved regression performance
✅ VIF-based feature reduction simplified models without major performance loss
✅ Final comparison table & visualization plots summarize all models
