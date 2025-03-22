from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import pandas as pd
import traceback

app = Flask(__name__)

# Load models and preprocessors
MODEL_PATH_CLASS = "models/classifier.pkl"
MODEL_PATH_REG = "models/regressor.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoders.pkl"
FEATURE_MAP_PATH = "models/feature_mappings.pkl"

# Define categorical columns
categorical_cols = ["functional group", "A", "A'", "Bi", "B'"]

try:
    # Load models and preprocessors, with better error handling
    classifier = None
    if os.path.exists(MODEL_PATH_CLASS):
        with open(MODEL_PATH_CLASS, "rb") as f:
            classifier = pickle.load(f)
            if hasattr(classifier, 'feature_names_in_'):
                classifier_features = classifier.feature_names_in_.tolist()
                print(f"Classifier was trained on {len(classifier_features)} features")
            else:
                print("WARNING: Classifier does not have feature_names_in_ attribute")
                classifier_features = []
    else:
        print(f"Warning: Classifier model file not found at {MODEL_PATH_CLASS}")
        classifier_features = []

    regressor = None
    if os.path.exists(MODEL_PATH_REG):
        with open(MODEL_PATH_REG, "rb") as f:
            regressor = pickle.load(f)
            if hasattr(regressor, 'feature_names_in_'):
                regressor_features = regressor.feature_names_in_.tolist()
                print(f"Regressor was trained on {len(regressor_features)} features")
            else:
                print("WARNING: Regressor does not have feature_names_in_ attribute")
                regressor_features = []
    else:
        print(f"Warning: Regressor model file not found at {MODEL_PATH_REG}")
        regressor_features = []

    scaler = None
    scaler_features = []
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = scaler.feature_names_in_.tolist()
                print(f"Scaler was trained on {len(scaler_features)} features")
            else:
                print("WARNING: Scaler does not have feature_names_in_ attribute")
    else:
        print(f"Warning: Scaler file not found at {SCALER_PATH}")

    label_encoders = {}
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f:
            label_encoders = pickle.load(f)
            print(f"Loaded encoders for {len(label_encoders)} categorical features")
    else:
        print(f"Warning: Label encoders file not found at {ENCODER_PATH}")
    
    feature_mappings = {"functional_group_mapping": {}, "ie_mapping": {}}
    if os.path.exists(FEATURE_MAP_PATH):
        with open(FEATURE_MAP_PATH, "rb") as f:
            feature_mappings = pickle.load(f)
            print(f"Loaded feature mappings for auto-filling missing values")
    else:
        print(f"Warning: Feature mappings file not found at {FEATURE_MAP_PATH}")

    functional_group_mapping = feature_mappings.get("functional_group_mapping", {})
    ie_mapping = feature_mappings.get("ie_mapping", {})
    
    # Define all expected features (combine from all sources)
    all_expected_features = set()
    for feature_list in [classifier_features, regressor_features, scaler_features]:
        if feature_list:
            all_expected_features.update(feature_list)
    
    all_expected_features = list(all_expected_features)
    print(f"Combined total of {len(all_expected_features)} unique expected features")
    
except Exception as e:
    print(f"Error loading models: {e}")
    traceback.print_exc()
    all_expected_features = []
    classifier = None
    regressor = None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Create a dataframe from the input data
        input_df = pd.DataFrame([data])
        
        # Auto-fill missing HOMO and LUMO based on IE+
        if "A_IE+" in data and "B_IE+" in data:
            key = (data["A_IE+"], data["B_IE+"])
            if key in ie_mapping:
                for key_name in ["A_HOMO-", "A_LUMO-", "B_HOMO-", "B_LUMO-"]:
                    if key_name not in data or pd.isna(data.get(key_name)):
                        input_df[key_name] = ie_mapping[key].get(key_name, 0)
        
        # Auto-fill missing anions and cations based on functional group
        if "functional group" in data and data["functional group"] in functional_group_mapping:
            for key in ["A", "A'", "Bi", "B'"]:
                if key not in data or pd.isna(data.get(key)):
                    input_df[key] = functional_group_mapping[data["functional group"]].get(key, "")
        
        # Process categorical features
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = input_df[col].astype(str)
                    # Check for unknown categories and replace them
                    for idx, val in enumerate(input_df[col]):
                        if val not in encoder.classes_:
                            print(f"Warning: Value '{val}' for '{col}' not in training data. Using default value.")
                            input_df.loc[idx, col] = encoder.classes_[0]
                    
                    input_df[col] = encoder.transform(input_df[col])
                except Exception as e:
                    print(f"Warning: Error encoding '{col}': {e}")
                    input_df[col] = 0
        
        # Handle numerical features with scaler if available
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            for col in scaler.feature_names_in_:
                if col in input_df.columns:
                    try:
                        # Reshape for single sample
                        col_data = np.array(input_df[col]).reshape(-1, 1)
                        # Transform just this column
                        input_df[col] = scaler.transform(col_data).flatten()
                    except Exception as e:
                        print(f"Warning: Error scaling '{col}': {e}")
        
        # Prepare feature sets that match what the models expect
        if classifier is not None and hasattr(classifier, 'feature_names_in_'):
            X_class = pd.DataFrame(index=[0])
            for feature in classifier.feature_names_in_:
                if feature in input_df.columns:
                    X_class[feature] = input_df[feature].values[0]
                else:
                    X_class[feature] = 0.0  # Default value for missing features
        else:
            X_class = input_df.copy()
            
        if regressor is not None and hasattr(regressor, 'feature_names_in_'):
            X_reg = pd.DataFrame(index=[0])
            for feature in regressor.feature_names_in_:
                if feature in input_df.columns:
                    X_reg[feature] = input_df[feature].values[0]
                else:
                    X_reg[feature] = 0.0  # Default value for missing features
        else:
            X_reg = input_df.copy()
        
        # Get classification result
        is_insulator = 1  # Default to insulator
        real_classification = "Insulator (default)"
        orig_classification = None
        
        try:
            if classifier is not None:
                orig_classification = classifier.predict(X_class)[0]
                is_insulator = orig_classification
                real_classification = "Insulator" if is_insulator == 1 else "Conductor"
                print(f"Classification result: {real_classification}")
            else:
                print("Warning: No classifier model available, using default classification")
        except Exception as e:
            print(f"Classification error: {e}")
            traceback.print_exc()
        
        # Calculate band gap prediction if it's an insulator
        band_gap = None
        predicted_gap = None
        
        try:
            if regressor is not None:
                predicted_gap = regressor.predict(X_reg)[0]
                print(f"Raw predicted band gap: {predicted_gap}")
                
                # Only set band_gap if it's an insulator
                if is_insulator == 1:
                    band_gap = predicted_gap
                    
                    # Validate the band gap prediction
                    if band_gap < 0.5:
                        print(f"Warning: Predicted band gap ({band_gap}) < 0.5 eV but classified as insulator")
                        if band_gap < 0.1:  # If extremely low, use default
                            band_gap = 0.5  # Minimum threshold for insulators
            else:
                print("Warning: No regressor model available")
                if is_insulator == 1:
                    band_gap = 0.5  # Default minimum for insulators
        except Exception as e:
            print(f"Regression error: {e}")
            traceback.print_exc()
            if is_insulator == 1:
                band_gap = 0.5  # Default minimum
        
        return jsonify({
            "classification": real_classification,
            "predicted_band_gap": float(band_gap) if band_gap is not None else None,
            "raw_classification": int(is_insulator) if is_insulator is not None else 1,
            "raw_predicted_band_gap": float(predicted_gap) if predicted_gap is not None else None
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    # Check if the required models are available before starting
    if not os.path.exists(MODEL_PATH_CLASS):
        print(f"Warning: Classifier model not found at {MODEL_PATH_CLASS}")
    if not os.path.exists(MODEL_PATH_REG):
        print(f"Warning: Regressor model not found at {MODEL_PATH_REG}")
    
    print("Starting Flask API server...")
    app.run(host="0.0.0.0", port=5000, debug=True)