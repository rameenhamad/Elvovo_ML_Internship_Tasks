# Elvovo_ML_Internship_Tasks

## Task: Student Exam Score Prediction
### Objective:
Predict student exam scores based on study habits, socio-economic factors, and school-related features.
### Dataset:
6607 records, **20 features** (hours studied, attendance, parental involvement, family income, teacher quality, etc.).
**Target variable:** Exam_Score (numeric).
### Preprocessing:
Handled **missing values** (Teacher_Quality, Parental_Education_Level, Distance_from_Home) with mode imputation.
### Feature engineering:
- Ordinal Encoding (e.g., Low–Medium–High, Peer Influence).
- One-Hot Encoding (e.g., Gender, Internet Access).
- Normalized using RobustScaler.
- Applied log transformation on target (Exam_Score) to stabilize distribution.
### Models Tried:
**Linear Regression**
MSE: 0.00043; RMSE: 0.0208; MAE: 0.0060; R²: **0.84** **Best**
**Polynomial Regression + Ridge**
MSE: 0.00047; RMSE: 0.0217; MAE: 0.0076; R²: **0.82**
**Polynomial Regression + Lasso**
MSE: 0.00264; RMSE: 0.0514; MAE: 0.0396; R²: ~-2 (**poor fit**)
### Observations:
- Linear Regression performed best (lowest error, highest R²).
- Ridge was close but unnecessary (regularization didn’t help).

## Task 2: Customer Segmentation
### Dataset: 
Mall Customers dataset (**200 entries**) with features: Genre, Age, Income, and Spending Score.
### Preprocessing: 
Encoded Gender, selected features, and checked data quality.
### Optimal Clusters: 
**Elbow Method** indicated **k=5** as the best fit.
### Clustering: 
Applied K-Means and grouped customers into 5 clusters.
### Visualization: 
Plotted Annual Income vs. Spending Score with cluster colors.
### Results: 
**Segments** identified as Luxury, Loyal, Potential, Average, Savers.
- **Luxury** & **Loyal** are profitable, **Potential** can grow, **Savers** need engagement.

## Task 3: Forest Cover Type Prediction
### Dataset: 
**Covertype dataset** (UCI, ~581k rows, 54 features + Cover_Type label).
### Preprocessing: 
- Removed duplicates
- Scaled numeric features (e.g., Elevation, Slope, Distances, Hillshade)
- Handled categorical Wilderness & Soil types.
### Models: 
Trained **Random_Forest** (200 trees) and **XGBoost** (200 estimators) for **multi-class** classification.
### Performance:
- **Random-Forest** → **96% accuracy**, stronger across most cover types.
- **XGBoost** → **91% accuracy**, slightly lower but faster training & prediction.
### Visualization: 
Compared confusion matrices for both models and plotted top **20 feature importances** (elevation, hillshade, distances most important).
- **Random-Forest** achieved the best **accuracy** but is **slower**; **XGBoost** is more **efficient** for real-time use.
Lasso severely underfit due to over-penalization.

Scatter plots confirmed Linear Regression predictions align closely with actual values.
