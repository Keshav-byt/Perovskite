import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data_path = "data/perovskite_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# Identify categorical and numerical columns
categorical_cols = ["functional group", "A", "A'", "Bi", "B'"]  # Modify if needed
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["PBE band gap"]]

# Encode categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert to numerical
    encoders[col] = le  # Store encoder for later use

# Save label encoders
os.makedirs("models", exist_ok=True)
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Standardize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Define target variable
X = df.drop(columns=["PBE band gap"])
y = df["PBE band gap"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
regressor.fit(X_train, y_train)

# Evaluate model
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\n MAE: {mae:.3f}, MSE: {mse:.3f}, R²: {r2:.3f}")

# Save trained model
with open("models/regressor.pkl", "wb") as f:
    pickle.dump(regressor, f)

print("Regressor model saved successfully!")