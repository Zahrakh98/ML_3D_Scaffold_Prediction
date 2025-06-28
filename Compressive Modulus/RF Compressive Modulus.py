import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, mean_squared_log_error
)
from numpy import expm1, log1p

# === Load data ===
df = pd.read_csv("data_compressive.csv", sep=';')

y_log = log1p(df['Compressive Modulus'])
X = df.drop(columns=['Compressive Modulus'])

# === Identify feature types ===
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# === Preprocessing ===
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# === Pipeline ===
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# === Hyperparameter grid ===
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 5, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# === Train/test split ===
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# === Grid search (no outer CV) ===
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)
grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train_log)

# === Final model ===
final_model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)

# === Predict on train and test ===
y_pred_train_log = final_model.predict(X_train)
y_pred_test_log = final_model.predict(X_test)

# Inverse transform
y_train = expm1(y_train_log)
y_test = expm1(y_test_log)
y_pred_train = expm1(y_pred_train_log)
y_pred_test = expm1(y_pred_test_log)

# === Adjusted R² ===
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

# === Evaluation ===
def evaluate_metrics(y_true, y_pred, X_sample, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    p = final_model.named_steps['preprocessing'].transform(X_sample).shape[1]
    n = len(y_true)
    adj_r2 = adjusted_r2(r2, n, p)

    print(f"\n{label} Set:")
    print(f"R²           : {r2:.4f}")
    print(f"Adjusted R²  : {adj_r2:.4f}")
    print(f"MAE          : {mae:.2f}")
    print(f"MSE          : {mse:.2f}")
    print(f"RMSE         : {rmse:.2f}")
    print(f"RMSLE        : {rmsle:.4f}")
    print(f"MAPE         : {mape:.2f}%")

evaluate_metrics(y_train, y_pred_train, X_train, label="Train")
evaluate_metrics(y_test, y_pred_test, X_test, label="Test")
