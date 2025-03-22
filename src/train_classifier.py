import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data_path = "data/perovskite_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# Define categorical and numerical columns
categorical_cols = ["functional group", "A", "A'", "Bi", "B'"]  # Update if needed
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["PBE band gap"]]

# Encode categorical features
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

# **Define classification target** (Binary classification: Insulator = 1, Conductor = 0)
df["classification_target"] = (df["PBE band gap"] >= 0.5).astype(int)

# Prepare feature-target split
X = df.drop(columns=["PBE band gap", "classification_target"])
y = df["classification_target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (Random Forest)
classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
classifier.fit(X_train, y_train)

# Evaluate model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

print("Classifier model saved successfully!")