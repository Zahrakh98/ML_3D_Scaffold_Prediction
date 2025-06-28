import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,  Dropout
from keras.callbacks import EarlyStopping
import os
import random

# Data preparation
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

df=pd.read_csv(r"data_cell_viability", sep=';')

#Define features and target
X = df.drop(['Cell viability'], axis=1)
y = df['Cell viability']

# === Step 2: Identify column types ===
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
# Preprocessing
# Split raw data before preprocessing
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=SEED)


# Fit preprocessor only on training data
preprocessor = ColumnTransformer(transformers=[
    ('num', RobustScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
# Apply transformations correctly
X_train = preprocessor.fit_transform(X_train)  # fit and transform on training
X_test = preprocessor.transform(X_test)        # only transform on test

# Neural Network model

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Training

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
y_train_pred = (model.predict(X_train).flatten())
y_test_pred = (model.predict(X_test).flatten())
y_train_true = (y_train)
y_test_true = (y_test)

# Train metrics
r2_train = r2_score(y_train_true, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
mae_train = mean_absolute_error(y_train_true, y_train_pred)

# Test metrics
r2_test = r2_score(y_test_true, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
mae_test = mean_absolute_error(y_test_true, y_test_pred)

print("\n------ Training Set Metrics ------")
print(f"Train R²: {r2_train:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Train MAE: {mae_train:.3f}")

print("\n------ Test Set Metrics ------")
print(f"Test R²: {r2_test:.3f}")
print(f"Test RMSE: {rmse_test:.3f}")
print(f"Test MAE: {mae_test:.3f}")

