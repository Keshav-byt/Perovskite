import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Check if preprocessed data exists
if os.path.exists("data/preprocessed.npz"):
    # Load preprocessed data
    data = np.load("data/preprocessed.npz", allow_pickle=True)
    X_train = data["X_train_class"]
    X_test = data["X_test_class"]
    y_train = data["y_train_class"]
    y_test = data["y_test_class"]
    
    # Ensure X data is correctly formatted (as DataFrame if it was saved as such)
    if isinstance(X_train, np.ndarray) and not isinstance(X_train, pd.DataFrame):
        try:
            # Try to convert to DataFrame if feature names are available
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train = pd.DataFrame(X_train, columns=feature_names)
            X_test = pd.DataFrame(X_test, columns=feature_names)
        except Exception as e:
            print(f"Warning: Could not convert array to DataFrame: {e}")
else:
    # If preprocessed data doesn't exist, run the preprocessing steps
    # Load dataset
    data_path = "data/perovskite_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Define categorical and numerical columns
    categorical_cols = ["functional group", "A", "A'", "Bi", "B'"]
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ["PBE band gap"]]

    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Save label encoders
    os.makedirs("models", exist_ok=True)
    with open("models/label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Standardize numerical features
    numerical_features_present = [col for col in numerical_cols if col in df.columns]
    if numerical_features_present:
        scaler = StandardScaler()
        scaler.feature_names_in_ = np.array(numerical_features_present)
        df[numerical_features_present] = scaler.fit_transform(df[numerical_features_present])

        # Save the scaler
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    # Define classification target (Binary classification: Insulator = 1, Conductor = 0)
    if "PBE band gap" in df.columns:
        df["classification_target"] = (df["PBE band gap"] >= 0.5).astype(int)

        # Prepare feature-target split
        X = df.drop(columns=["PBE band gap", "classification_target"])
        y = df["classification_target"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        raise ValueError("PBE band gap column not found in dataset")

# Train a classifier (Random Forest)
classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
classifier.fit(X_train, y_train)
# Store feature names
if hasattr(X_train, 'columns'):
    classifier.feature_names_in_ = np.array(X_train.columns)

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