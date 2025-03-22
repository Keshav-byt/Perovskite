import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained classification model and preprocessed test data
with open("models/classification_model.pkl", "rb") as f:
    clf = pickle.load(f)

data = np.load("data/preprocessed.npz", allow_pickle=True)
X_test_class = data['X_test_class']
y_test_class = data['y_test_class']

# Make predictions
y_pred_class = clf.predict(X_test_class)

# Generate confusion matrix and classification report
cm = confusion_matrix(y_test_class, y_pred_class)
print("Classification Report:\n", classification_report(y_test_class, y_pred_class))

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Insulator", "Insulator"], yticklabels=["Non-Insulator", "Insulator"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Classification Model")
plt.show()

# Feature importance for Logistic Regression using coefficients
coefficients = clf.coef_[0]
if hasattr(X_test_class, "columns"):
    feature_names = X_test_class.columns
else:
    feature_names = [f"feature_{i}" for i in range(X_test_class.shape[1])]

# Zip features and coefficients together and sort by the absolute value of the coefficients
importance_df = np.array(list(zip(feature_names, coefficients)))
importance_df = importance_df[np.argsort(np.abs(coefficients))[::-1]]  # sort by magnitude

print("Top 5 features by absolute coefficient value:")
for feat, coef in importance_df[:5]:
    print(f"{feat}: {float(coef):.3f}")