# Perovskite Band Gap Prediction  

This project builds a machine learning model to predict the band gap of perovskite materials. The workflow includes data preprocessing, model training, evaluation, and a Flask API for making predictions.  

## Features  

- **Preprocessing:** Cleans and encodes data for classification and regression tasks.  
- **Model Training:** Trains separate models for classification (insulator vs. conductor) and regression (predicting band gap).  
- **Evaluation:** Computes model performance metrics.  
- **API Deployment:** Provides a Flask-based API for predictions.  
- **Request Script:** Sends sample input to the API.  

---

## 1. Installation  

### Prerequisites  

- Python 3.8+  
- Virtual environment (optional but recommended)  

### Setup  

```bash
git clone https://github.com/your-repo/perovskite_bandgap_prediction.git  
cd perovskite_bandgap_prediction  
python -m venv .venv  
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -r requirements.txt  
```

---

## 2. Data Preprocessing  

Ensure your dataset (`perovskite_data.csv`) is placed in the `data/` directory. Then run:  

```bash
python src/preprocess.py  
```

This script:  

- Encodes categorical features.  
- Standardizes numerical features.  
- Saves processed data and encoders.  

---

## 3. Model Training  

Train classification and regression models:  

```bash
python src/train.py  
```

This script:  

- Loads preprocessed data.  
- Trains a **RandomForestClassifier** for classification.  
- Trains a **RandomForestRegressor** for regression.  
- Saves trained models in the `models/` directory.  

---

## 4. Model Evaluation  

Evaluate model performance:  

```bash
python src/evaluate.py  
```

This script:  

- Loads trained models.  
- Computes classification accuracy and regression MAE.  

---

## 5. API Deployment  

Start the Flask API:  

```bash
python src/main.py  
```

By default, the API runs on `http://127.0.0.1:5000`.  

### API Endpoints  

- **POST `/predict`**  
  - Input: JSON object with 37 feature values.  
  - Output: Predicted classification and regression values.  

---

## 6. Making Predictions  

Use the provided script to send a sample request:  

```bash
python request.py  
```

---

## 7. Project Structure  

```
perovskite_bandgap_prediction/
│── data/                     # Contains raw and preprocessed data  
│── models/                   # Stores trained models and encoders  
│── src/  
│   ├── preprocess.py         # Data preprocessing  
│   ├── train.py              # Model training  
│   ├── evaluate.py           # Model evaluation  
│   ├── main.py               # Flask API  
│── request.py                # Sample API request script  
│── requirements.txt          # Dependencies  
│── README.md                 # Project documentation  
```

---

## 8. Notes  

- Ensure the dataset matches expected column names.  
- Preprocessing must be run before training.  
- Encoders must be used consistently in preprocessing and inference.  

---

## 9. Future Improvements  

- Try different machine learning models.  
- Optimize hyperparameters.  
- Extend API for batch predictions.  

---

This project enables automated band gap prediction using machine learning. Follow the steps above to preprocess, train, and deploy the models.
