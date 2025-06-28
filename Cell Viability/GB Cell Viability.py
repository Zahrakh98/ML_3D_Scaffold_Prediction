import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from math import sqrt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df=pd.read_csv(r"data_cell_viability", sep=';')

#Define features and target
X = df.drop(['Cell viability'], axis=1)
y = df['Cell viability']
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# --- Preprocessing ---
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Cross-Validation Setup
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# Initialize Gradient Boosting Regressor with GridSearchCV

gb_cv = GridSearchCV(
    estimator=pipeline, 
    param_grid={
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.5],
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    },
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)
gb_cv.fit(X_train, y_train)

# Best model
best_gb = gb_cv.best_estimator_
y_train_pred = best_gb.predict(X_train)
y_test_pred = best_gb.predict(X_test)

# Train and Test R²
y_train_pred = best_gb.predict(X_train)
y_test_pred = best_gb.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Cross-Validation Results
cv_scores = gb_cv.cv_results_['mean_test_score']
# Print Best Hyperparameters
print(f"Best Hyperparameters: {gb_cv.best_params_}")

print(f"\nTrain R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Mean Cross-Validated R²: {np.mean(cv_scores):.4f}")

# Error Calculation
print(f"Mean Absolute Error (Train): {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"Mean Absolute Error (Test): {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"Mean Squared Error (Train): {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Mean Squared Error (Test): {mean_squared_error(y_test, y_test_pred):.4f}")


