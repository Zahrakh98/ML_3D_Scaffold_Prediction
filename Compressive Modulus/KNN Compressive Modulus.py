import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os, random
import tensorflow as tf

# Set seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

df=pd.read_csv(r"data_compressive.csv", sep=';')


X = df.drop(['Compressive Modulus'], axis=1)
y = np.log1p(df["Compressive Modulus"])  # Log-transform the target

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
# Preprocessing

preprocessor = ColumnTransformer([
    ('num', PowerTransformer(method='yeo-johnson', standardize=True), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# KNN Regressor

knn = KNeighborsRegressor()

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', knn)
])

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)


# Hyperparameter Tuning

param_grid = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11],
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


# Evaluation

# Predict and inverse transform
log_y_train_pred = best_model.predict(X_train)
log_y_test_pred = best_model.predict(X_test)

y_train_pred = np.expm1(log_y_train_pred)
y_test_pred = np.expm1(log_y_test_pred)

y_train_true = np.expm1(y_train)
y_test_true = np.expm1(y_test)

# Train metrics
r2_train = r2_score(y_train_true, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
mae_train = mean_absolute_error(y_train_true, y_train_pred)

# Test metrics
r2_test = r2_score(y_test_true, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
mae_test = mean_absolute_error(y_test_true, y_test_pred)

# Results

print("Best Parameters:", grid_search.best_params_)

print("\n------ Training Set Metrics ------")
print(f"Train R²: {r2_train:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Train MAE: {mae_train:.3f}")

print("\n------ Test Set Metrics ------")
print(f"Test R²: {r2_test:.3f}")
print(f"Test RMSE: {rmse_test:.3f}")
print(f"Test MAE: {mae_test:.3f}")


