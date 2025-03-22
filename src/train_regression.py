import os
import numpy as np
import pickle
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load preprocessed data for regression
data_path = "data/preprocessed.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Preprocessed data file not found at {data_path}")

data = np.load(data_path, allow_pickle=True)
X_train_reg = data['X_train_reg']
X_test_reg  = data['X_test_reg']
y_train_reg = data['y_train_reg']
y_test_reg  = data['y_test_reg']

# Initialize and train the Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg, y_train_reg)

# Predict on the test set for regression
y_pred_reg = regressor.predict(X_test_reg)

# Evaluate the regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("Regression Model Performance:")
print(f"RMSE : {rmse:.3f}")
print(f"R^2  : {r2:.3f}")

# Save the trained regression model
os.makedirs("models", exist_ok=True)
with open("models/regression_model.pkl", "wb") as f:
    pickle.dump(regressor, f)

print("Regression model training complete! Model saved to 'models/regression_model.pkl'")
