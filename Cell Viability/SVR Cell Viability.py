
# SVR Regression: Cell Viability

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv(r"data_cell_viability", sep=';')

#Define features and target
X = df.drop(['Cell viability'], axis=1)
y = df['Cell viability']

# === Step 2: Identify column types ===
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_cols), 
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

svr = SVR()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', svr)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter Tuning

param_grid = {
    'regressor__C': [1, 10, 50, 100, 200],
    'regressor__epsilon': [0.001, 0.01, 0.05, 0.1],
    'regressor__gamma': ['scale', 'auto', 0.01, 0.1],
    'regressor__kernel': ['rbf']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
                           

grid_search.fit(X_train, y_train)

# Evaluation

best_model = grid_search.best_estimator_

# --- Test set ---
y_pred = best_model.predict(X_test)


r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# --- Train set ---
y_train_pred = best_model.predict(X_train)
y_train_true = y_train

r2_train = r2_score(y_train_true, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
mae_train = mean_absolute_error(y_train_true, y_train_pred)

# Print Results
print("Best Parameters:", grid_search.best_params_)
print("Training Set Metrics")
print(f"Train R²: {r2_train:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Train MAE: {mae_train:.3f}")

print("Test Set Metrics")
print(f"Test R²: {r2:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE: {mae:.3f}")

