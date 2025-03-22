from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import pandas as pd
import traceback

app = Flask(__name__)

# Load models and preprocessors
MODEL_PATH_CLASS = "models/classifier.pkl"  # Use the balanced model
MODEL_PATH_REG = "models/regressor.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoders.pkl"

try:
    # Use balanced classifier if available, fall back to original
    if os.path.exists(MODEL_PATH_CLASS):
        with open(MODEL_PATH_CLASS, "rb") as f:
            classifier = pickle.load(f)

        with open(MODEL_PATH_REG, "rb") as f:
            regressor = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        with open(ENCODER_PATH, "rb") as f:
            label_encoders = pickle.load(f)
    
    # Load a sample of training data to get feature names
    data = np.load("data/preprocessed.npz", allow_pickle=True)
    X_train_class = data['X_train_class']
    if isinstance(X_train_class, pd.DataFrame):
        FEATURE_NAMES = X_train_class.columns.tolist()
    else:
        FEATURE_NAMES = None
        print("ERROR: Could not determine feature names")
except Exception as e:
    print(f"Error loading models: {e}")
    traceback.print_exc()

@app.route("/predict", methods=["POST"])
def predict():
    print("Received prediction request")
    data = request.json
    print(f"Input data: {data}")
    
    try:
        # Create a dataframe from the input data
        input_df = pd.DataFrame([data])
        
        # 1. Process categorical features
        print("Processing categorical features...")
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except Exception as e:
                    print(f"Warning: Error encoding '{col}': {e}")
                    # Use first class in encoder as fallback
                    input_df[col] = 0
        
        # 2. Handle feature alignment
        print("Aligning features...")
        if FEATURE_NAMES:
            # Add missing columns
            for col in FEATURE_NAMES:
                if col not in input_df.columns:
                    print(f"Adding missing column: {col}")
                    input_df[col] = 0
            
            # Select only needed columns in correct order
            X = input_df[FEATURE_NAMES]
        else:
            # If we couldn't determine feature names, just hope the order is correct
            X = input_df.drop(columns=['pbe band gap', 'insulator'], errors='ignore')
        
        print(f"Feature count for prediction: {X.shape[1]}")
        
        # 3. Make classification prediction
        print("Making classification prediction...")
        is_insulator = classifier.predict(X)[0]
        print(f"Classification result: {is_insulator} ({'Insulator' if is_insulator == 1 else 'Conductor'})")
        
        # 4. Make regression prediction if insulator
        band_gap = None
        if is_insulator == 1:
            print("Making regression prediction for band gap...")
            band_gap = regressor.predict(X)[0]
            print(f"Predicted band gap: {band_gap}")
        
        return jsonify({
            "classification": "Insulator" if is_insulator == 1 else "Conductor",
            "predicted_band_gap": float(band_gap) if band_gap is not None else None,
            }
        )
    
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask API server...")
    app.run(host="0.0.0.0", port=5000, debug=True)