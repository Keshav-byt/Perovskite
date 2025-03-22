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

# Optional: Inspect the first few rows of the dataset
print("Initial data preview:")
print(df.head())

# Check for missing values and impute if necessary
if df.isnull().values.any():
    print("Missing values detected. Imputing missing values...")
    # For numerical columns, fill missing values with the median
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill missing values with the mode
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
else:
    print("No missing values detected.")

# Define categorical and numerical columns.
# Make sure these match your dataset exactly.
categorical_cols = ["functional group", "A", "A'", "Bi", "B'"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["PBE band gap"]]

# Encode categorical variables using LabelEncoder and store the encoders
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save encoder for future transformations

# Save label encoders for later use
os.makedirs("models", exist_ok=True)
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Standardize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the scaler for future use
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Optional: Inspect the preprocessed data
print("Data after preprocessing:")
print(df.head())

# Split data for classification and regression tasks
# For classification: target is whether PBE band gap is >= 0.5 eV
X = df.drop(columns=["PBE band gap"])
y_class = (df["PBE band gap"] >= 0.5).astype(int)

# Create train-test split for classification (80/20 split)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# For regression: use only insulators (PBE band gap >= 0.5 eV)
insulators = df[df["PBE band gap"] >= 0.5]
X_reg = insulators.drop(columns=["PBE band gap"])
y_reg = insulators["PBE band gap"]

# Create train-test split for regression (80/20 split)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Save preprocessed data to a .npz file for later use
np.savez("data/preprocessed.npz",
         X_train_class=X_train_class, X_test_class=X_test_class, 
         y_train_class=y_train_class, y_test_class=y_test_class,
         X_train_reg=X_train_reg, X_test_reg=X_test_reg, 
         y_train_reg=y_train_reg, y_test_reg=y_test_reg)

print("Preprocessing complete! Data saved to 'data/preprocessed.npz'")