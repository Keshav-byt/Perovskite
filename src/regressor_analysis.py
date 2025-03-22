import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained regression model and preprocessed test data for regression
with open("models/regression_model.pkl", "rb") as f:
    regressor = pickle.load(f)

data = np.load("data/preprocessed.npz", allow_pickle=True)
X_test_reg = data['X_test_reg']
y_test_reg = data['y_test_reg']

# Make predictions
y_pred_reg = regressor.predict(X_test_reg)

# Compute residuals and evaluate performance
residuals = y_test_reg - y_pred_reg
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Regression Performance:\nMSE: {mse:.3f}, R²: {r2:.3f}")

# Plot residual distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residual Distribution for Regression Model")
plt.xlabel("Residual (True - Predicted)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot of predictions vs. true values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], "r--")
plt.xlabel("True Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title("Predicted vs. True Band Gap Values")
plt.show()

# Feature importance for Random Forest Regressor
importances = regressor.feature_importances_
if hasattr(X_test_reg, "columns"):
    feature_names = X_test_reg.columns
else:
    feature_names = [f"feature_{i}" for i in range(X_test_reg.shape[1])]

importance_df = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("Top 5 features by importance (Regression):")
for feat, imp in importance_df[:5]:
    print(f"{feat}: {imp:.3f}")

# Bar plot for feature importances
feat_names, feat_importances = zip(*importance_df[:10])  # top 10 for visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=list(feat_importances), y=list(feat_names))
plt.title("Top 10 Feature Importances for Regression Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
