from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from numpy import expm1, log1p


df=pd.read_csv(r"data_compressive.csv", sep=';')

y_log = log1p(df['Compressive Modulus'])  # log-transform target
X = df.drop(['Compressive Modulus'], axis=1)

# Separate features by type
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

#Define preprocessing and model pipeline 
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# === Hyperparameter grid ===
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.1, 0.5],
    'regressor__max_depth': [3, 5, 7, 10],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

#Nested Cross-Validation Setup
outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)

# === Store metrics ===
metrics = {'r2': [], 'mae': [], 'rmse': []}


# Nested CV Loop
for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_log, y_test_log = y_log.iloc[train_idx], y_log.iloc[test_idx]

    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train_log)

    best_model = grid_search.best_estimator_
    y_pred_log = best_model.predict(X_test)

    # Log-space metrics (optional)
    r2_log = r2_score(y_test_log, y_pred_log)
    mae_log = mean_absolute_error(y_test_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
    metrics['r2'].append(r2_log)
    metrics['mae'].append(mae_log)
    metrics['rmse'].append(rmse_log)

# Summary function
def summarize(metric_dict, label):
    print(f"\n✅ Mean ± Std ({label}):")
    for k, v in metric_dict.items():
        v = np.array(v)
        print(f"{k.upper():<6}: {v.mean():.4f} ± {v.std():.4f}")

# === Final summary of Nested CV ===
summarize(metrics, "CV")


# === Final 80/20 Train-Test Evaluation (Original Scale Metrics) ===
print("Train-Test Evaluation (Original Scale):")

# Split the full dataset
X_train_final, X_test_final, y_train_log_final, y_test_log_final = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

#Fit GridSearchCV on training set
grid_search_final = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='r2', n_jobs=-1)
grid_search_final.fit(X_train_final, y_train_log_final)
final_model = grid_search_final.best_estimator_

print("Best hyperparameters:", grid_search_final.best_params_)
#  Predict on train and test
y_pred_train_log = final_model.predict(X_train_final)
y_pred_test_log = final_model.predict(X_test_final)

# Inverse transform predictions and targets
y_train_final = expm1(y_train_log_final)
y_test_final = expm1(y_test_log_final)
y_pred_train = expm1(y_pred_train_log)
y_pred_test = expm1(y_pred_test_log)

def adjusted_r2(r2, n, p):
    if n > p + 1:
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        return np.nan

# Metrics for Train and Test (Original Scale)
def evaluate_metrics(y_true, y_pred, X_sample, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Use transformed data to get actual number of features after one-hot
    p = final_model.named_steps['preprocessor'].transform(X_sample).shape[1]
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




