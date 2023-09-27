import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import h5py
import joblib

# Load the trained model
model = joblib.load('models/fraud_detection_model.pkl')

# Load a new dataset for evaluation
X_new = np.load("data/X_new.npy")
y_new = np.load("data/y_new.npy")

# Make predictions using the loaded model
y_pred = model.predict(X_new)

# Evaluate the model on the new dataset
accuracy = accuracy_score(y_new, y_pred)
report = classification_report(y_new, y_pred)
conf_matrix = confusion_matrix(y_new, y_pred)

print(f"Accuracy: {accuracy}")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

joblib.dump(model,'models/new_fraud_detection_model.pkl')