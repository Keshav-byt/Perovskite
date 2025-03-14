import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

def load_preprocessed_data():
    """Load preprocessed data from file."""
    data_path = "data/preprocessed.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Preprocessed data not found. Run preprocess.py first.")
    
    data = np.load(data_path, allow_pickle=True)
    return (data['X_train_class'], data['X_test_class'], data['y_train_class'], data['y_test_class'],
            data['X_train_reg'], data['X_test_reg'], data['y_train_reg'], data['y_test_reg'])

def train_classification_model(X_train, y_train):
    """Train and save a RandomForest classification model."""
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Empty dataset: Classification model cannot be trained.")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf

def fix_classifier():
    """Retrain the classifier with class balancing to fix bias toward conductors"""
    
    try:
        # Load preprocessed data
        data = np.load("data/preprocessed.npz", allow_pickle=True)
        X_train_class = data['X_train_class']
        y_train_class = data['y_train_class']
        
        print(f"Training data shape: {X_train_class.shape}")
        print(f"Class distribution: {np.bincount(y_train_class)}")
        
        # Check class imbalance
        n_conductors = np.sum(y_train_class == 0)
        n_insulators = np.sum(y_train_class == 1)
        
        if n_insulators == 0:
            print("ERROR: No insulators in training data! Can't train a useful model.")
            return
        
        class_ratio = n_conductors / n_insulators
        print(f"Class imbalance ratio (conductors:insulators): {class_ratio:.2f}")
        
        clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'  # Use balanced class weights
        )
        clf.fit(X_train_class, y_train_class)
        
        # Verify model now predicts some insulators
        y_pred = clf.predict(X_train_class)
        predicted_insulators = np.sum(y_pred == 1)
        
        # Save improved model
        os.makedirs("models", exist_ok=True)
        with open("models/classifier.pkl", "wb") as f:
            pickle.dump(clf, f)
        
    except Exception as e:
        print(f"Error fixing classifier: {e}")

def train_regression_model(X_train, y_train):
    """Train and save a RandomForest regression model."""
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Empty dataset: Regression model cannot be trained.")
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    with open("models/regressor.pkl", "wb") as f:
        pickle.dump(reg, f)
    
    return reg

def evaluate_models(clf, X_test_class, y_test_class, reg, X_test_reg, y_test_reg):
    """Evaluate the classification and regression models."""
    y_pred_class = clf.predict(X_test_class)
    class_acc = accuracy_score(y_test_class, y_pred_class)
    print(f"Classification Model Accuracy: {class_acc:.4f}")
    
    y_pred_reg = reg.predict(X_test_reg)
    reg_mae = mean_absolute_error(y_test_reg, y_pred_reg)
    print(f"Regression Model MAE: {reg_mae:.4f}")

if __name__ == "__main__":
    fix_classifier()
    try:
        print("Loading preprocessed data...")
        X_train_class, X_test_class, y_train_class, y_test_class, X_train_reg, X_test_reg, y_train_reg, y_test_reg = load_preprocessed_data()
        
        print(f"Classification Data: X_train: {X_train_class.shape}, X_test: {X_test_class.shape}")
        print(f"Regression Data: X_train: {X_train_reg.shape}, X_test: {X_test_reg.shape}")
        
        print("Training classification model...")
        classifier = train_classification_model(X_train_class, y_train_class)
        
        print("Training regression model...")
        regressor = train_regression_model(X_train_reg, y_train_reg)
        
        print("Evaluating models...")
        evaluate_models(classifier, X_test_class, y_test_class, regressor, X_test_reg, y_test_reg)
        
        print("Training complete.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Data Error: {e}")
