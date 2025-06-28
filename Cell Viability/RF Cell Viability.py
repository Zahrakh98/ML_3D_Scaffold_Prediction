import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, mean_squared_log_error
)
from numpy import sqrt
df=pd.read_csv(r"data_cell_viability", sep=';')
# Define features and target
X = df.drop(['Cell viability'], axis=1)
y = df['Cell viability']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

#  Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

#Hyperparameter grid 
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 5, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Bin target for stratification 
bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_binned = bins.fit_transform(y.values.reshape(-1, 1)).astype(int).flatten()

# (Stratified Outer CV)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)

metrics = {'r2': [], 'mae': [], 'rmse': [], 'mape': []}

for train_idx, test_idx in outer_cv.split(X, y_binned):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    metrics['r2'].append(r2)
    metrics['mae'].append(mae)
    metrics['rmse'].append(rmse)
    metrics['mape'].append(mape)

#Summary of outer CV
def summarize(metric_dict, label):
    print(f"\n✅ Mean ± Std ({label}):")
    for k, v in metric_dict.items():
        v = np.array(v)
        print(f"{k.upper():<6}: {v.mean():.4f} ± {v.std():.4f}")

summarize(metrics, "CV")

# === Final 80/20 Split Evaluation ===
print("Train-Test Evaluation (Original Scale):")

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid_final = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='r2', n_jobs=-1)
grid_final.fit(X_train_final, y_train_final)
final_model = grid_final.best_estimator_

# Predict
y_pred_train = final_model.predict(X_train_final)
y_pred_test = final_model.predict(X_test_final)

def adjusted_r2(r2, n, p):
    if n > p + 1:
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        return np.nan
# Evaluate function
def evaluate_metrics(y_true, y_pred, X_sample, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    p = final_model.named_steps['preprocessing'].transform(X_sample).shape[1]

    n = len(y_true)
    adj_r2 = adjusted_r2(r2, n, p)


    print(f"{label} Set:")
    print(f"R²           : {r2:.4f}")
    print(f"Adjusted R²  : {adj_r2:.4f}")
    print(f"MAE          : {mae:.2f}")
    print(f"MSE          : {mse:.2f}")
    print(f"RMSE         : {rmse:.2f}")
    print(f"RMSLE        : {rmsle:.4f}")
    print(f"MAPE         : {mape:.2f}%")

evaluate_metrics(y_train_final, y_pred_train, X_train_final, label="Train")
evaluate_metrics(y_test_final, y_pred_test, X_test_final, label="Test")
