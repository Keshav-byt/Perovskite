import os
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data for classification
data_path = "data/preprocessed.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Preprocessed data file not found at {data_path}")

data = np.load(data_path, allow_pickle=True)
X_train_class = data['X_train_class']
X_test_class  = data['X_test_class']
y_train_class = data['y_train_class']
y_test_class  = data['y_test_class']

# Initialize and train the Logistic Regression model
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_class, y_train_class)

# Predict on the test set
y_pred_class = clf.predict(X_test_class)

# Evaluate the classification model
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)

print("Classification Model Performance:")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")

# Save the trained classification model
os.makedirs("models", exist_ok=True)
with open("models/classification_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Classification model training complete! Model saved to 'models/classification_model.pkl'")
