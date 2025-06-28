# SVR Regression Pipeline: Compressive Modulus

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df=pd.read_csv(r"data_compressive.csv", sep=';')


X = df.drop(['Compressive Modulus'], axis=1)
y = np.log1p(df["Compressive Modulus"])  # Log-transform the target

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', PowerTransformer(method='yeo-johnson', standardize=True), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)
# SVR Model
svr = SVR()

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', svr)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Hyperparameter Tuning
param_grid = {
    'regressor__C': [10, 100, 500, 1000],
    'regressor__epsilon': [0.01, 0.1, 0.5],
    'regressor__gamma': ['scale', 0.01, 0.001],
    'regressor__kernel': ['rbf']
}


grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)                     

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Evaluation

best_model = grid_search.best_estimator_
log_y_pred = best_model.predict(X_test)
log_y_test = y_test

# Inverse transform back to the original scale
y_pred = np.expm1(log_y_pred)
y_test_original = np.expm1(log_y_test)

# Now compute metrics on original scale
r2 = r2_score(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
mae = mean_absolute_error(y_test_original, y_pred)

# --- Train Evaluation ---
log_y_train_pred = best_model.predict(X_train)
y_train_pred = np.expm1(log_y_train_pred)
y_train_true = np.expm1(y_train)

r2_train = r2_score(y_train_true, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
mae_train = mean_absolute_error(y_train_true, y_train_pred)

# --- Print ---
print("------ Training Set Metrics ------")
print(f"Train R²: {r2_train:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Train MAE: {mae_train:.3f}")

print("\n------ Test Set Metrics ------")
print(f"Test R²: {r2:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE: {mae:.3f}")


