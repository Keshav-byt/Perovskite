'''import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

def load_preprocessed_data():
    """Load preprocessed test data."""
    data_path = "data/preprocessed.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Preprocessed data not found. Run preprocess.py first.")
    
    data = np.load(data_path, allow_pickle=True)
    return (data['X_test_class'], data['y_test_class'], 
            data['X_test_reg'], data['y_test_reg'])

def load_model(model_path):
    """Load a trained model from a file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Train the model first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def evaluate_models():
    """Evaluate the classification and regression models."""
    try:
        print("Loading test data...")
        X_test_class, y_test_class, X_test_reg, y_test_reg = load_preprocessed_data()

        print("Loading trained models...")
        classifier = load_model("models/classifier.pkl")
        regressor = load_model("models/regressor.pkl")

        print("Evaluating classification model...")
        y_pred_class = classifier.predict(X_test_class)
        class_acc = accuracy_score(y_test_class, y_pred_class)
        print(f"Classification Model Accuracy: {class_acc:.4f}")

        print("Evaluating regression model...")
        y_pred_reg = regressor.predict(X_test_reg)
        reg_mae = mean_absolute_error(y_test_reg, y_pred_reg)
        print(f"Regression Model MAE: {reg_mae:.4f}")
    
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    evaluate_models()
'''