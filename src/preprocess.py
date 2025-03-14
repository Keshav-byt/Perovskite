import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

def load_data(file_path):
    """Load the dataset from a CSV file and check if it exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the file path.")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the dataset: encode categorical variables, scale numerical features, and split data."""
    df = df.copy()
    
    # Standardize column names (remove spaces, lowercase for consistency)
    df.columns = df.columns.str.strip().str.lower()
    
    # Define target variable for classification
    if 'pbe band gap' not in df.columns:
        raise KeyError("Column 'PBE band gap' is missing from the dataset.")
    df['insulator'] = (df['pbe band gap'] >= 0.5).astype(int)
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()  # Identify all categorical columns dynamically
    numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['pbe band gap', 'insulator']]
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to avoid errors
        label_encoders[col] = le  # Save encoder for later use
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Split dataset into classification task (all data) and regression task (only insulators)
    X_class = df.drop(columns=['pbe band gap', 'insulator'])
    y_class = df['insulator']
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    
    df_insulators = df[df['insulator'] == 1]
    X_reg = df_insulators.drop(columns=['pbe band gap', 'insulator'])
    y_reg = df_insulators['pbe band gap']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save preprocessors
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    
    return (X_train_class, X_test_class, y_train_class, y_test_class, 
            X_train_reg, X_test_reg, y_train_reg, y_test_reg)

if __name__ == "__main__":
    file_path = "data/perovskite_data.csv"
    
    try:
        df = load_data(file_path)
        print("Dataset loaded successfully! Columns:", df.columns.tolist())
        X_train_class, X_test_class, y_train_class, y_test_class, X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocess_data(df)
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Save preprocessed data
        np.savez("data/preprocessed.npz", 
                 X_train_class=X_train_class, X_test_class=X_test_class, y_train_class=y_train_class, y_test_class=y_test_class, 
                 X_train_reg=X_train_reg, X_test_reg=X_test_reg, y_train_reg=y_train_reg, y_test_reg=y_test_reg)
        print("Preprocessing completed and data saved!")
    except KeyError as e:
        print(f"Missing column error: {e}")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(f"Data processing error: {e}")
