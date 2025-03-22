import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check if preprocessed data exists
if os.path.exists("data/preprocessed.npz"):
    # Load preprocessed data
    data = np.load("data/preprocessed.npz", allow_pickle=True)
    X_train = data["X_train_reg"]
    X_test = data["X_test_reg"]
    y_train = data["y_train_reg"]
    y_test = data["y_test_reg"]
    
    # Ensure X data is correctly formatted
    if isinstance(X_train, np.ndarray) and not isinstance(X_train, pd.DataFrame):
        try:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train = pd.DataFrame(X_train, columns=feature_names)
            X_test = pd.DataFrame(X_test, columns=feature_names)
        except Exception as e:
            print(f"Warning: Could not convert array to DataFrame: {e}")
else:
    # Run preprocessing steps if data doesn't exist
    # Load dataset
    data_path = "data/perovskite_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Process data for regression (insulators only)
    if "PBE band gap" in df.columns:
        insulators = df[df["PBE band gap"] >= 0.5]
        X = insulators.drop(columns=["PBE band gap"])
        y = insulators["PBE band gap"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        raise ValueError("PBE band gap column not found in dataset")

# Create a simpler regressor - RandomForest for more robustness
print("Training a Random Forest Regressor model...")
regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
regressor.fit(X_train, y_train)

# Store feature names
if hasattr(X_train, 'columns'):
    regressor.feature_names_in_ = np.array(X_train.columns)

# Evaluate on test set
y_pred_rf = regressor.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Performance - MAE: {mae_rf:.4f}, MSE: {mse_rf:.4f}, RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

# Save Random Forest regressor
with open("models/regressor.pkl", "wb") as f:
    pickle.dump(regressor, f)

print("Random Forest Regressor saved successfully!")

# Create Deep Neural Network model as a backup option
def create_deep_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Convert dataframes to numpy for Keras
X_train_nn = X_train.values if hasattr(X_train, 'values') else X_train
X_test_nn = X_test.values if hasattr(X_test, 'values') else X_test
y_train_nn = y_train.values if hasattr(y_train, 'values') else y_train
y_test_nn = y_test.values if hasattr(y_test, 'values') else y_test

# Train deep neural network
nn_model = create_deep_nn_model(X_train_nn.shape[1])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
    ModelCheckpoint('models/deep_nn_model.h5', save_best_only=True)
]

try:
    history = nn_model.fit(
        X_train_nn, y_train_nn,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    y_pred_nn = nn_model.predict(X_test_nn).flatten()
    mae_nn = mean_absolute_error(y_test_nn, y_pred_nn)
    mse_nn = mean_squared_error(y_test_nn, y_pred_nn)
    rmse_nn = np.sqrt(mse_nn)
    r2_nn = r2_score(y_test_nn, y_pred_nn)

    print(f"Neural Network Performance - MAE: {mae_nn:.4f}, MSE: {mse_nn:.4f}, RMSE: {rmse_nn:.4f}, R²: {r2_nn:.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_nn, y_pred_nn, alpha=0.5)
    plt.plot([min(y_test_nn), max(y_test_nn)], [min(y_test_nn), max(y_test_nn)], 'r--')
    plt.title('Neural Network: Actual vs Predicted Band Gap Values')
    plt.xlabel('Actual Band Gap (eV)')
    plt.ylabel('Predicted Band Gap (eV)')
    plt.savefig('models/nn_prediction_comparison.png')
    plt.close()

    print("Neural network model training complete.")
except Exception as e:
    print(f"Neural network training failed: {e}")
    print("Continuing with Random Forest regressor only.")