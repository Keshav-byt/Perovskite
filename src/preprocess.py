import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load dataset
data_path = "data/perovskite_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

print("Dataset loaded successfully! Columns:", df.columns.tolist())

# Define categorical and numerical columns
categorical_cols = ["functional group", "A", "A'", "Bi", "B'"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["PBE band gap"]]

# Encode categorical variables
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
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

# Split data for classification and regression tasks
X = df.drop(columns=["PBE band gap"])
y_class = (df["PBE band gap"] >= 0.5).astype(int)  # Binary classification target

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Filter only insulators for regression task
insulators = df[df["PBE band gap"] >= 0.5]
X_reg = insulators.drop(columns=["PBE band gap"])
y_reg = insulators["PBE band gap"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Save preprocessed data
np.savez("data/preprocessed.npz",
         X_train_class=X_train_class, X_test_class=X_test_class, y_train_class=y_train_class, y_test_class=y_test_class,
         X_train_reg=X_train_reg, X_test_reg=X_test_reg, y_train_reg=y_train_reg, y_test_reg=y_test_reg)

print("Preprocessing complete! Data saved to 'data/preprocessed.npz'")
