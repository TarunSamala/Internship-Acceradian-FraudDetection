import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the cleaned dataset
data = pd.read_csv("data/cleaned_dataset.csv")

# Split the data into features (X) and target (y)
X = data.drop(columns=['isFraud'])
y = data['isFraud']

# Step 1: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define data preprocessing steps using ColumnTransformer and Pipeline
# Specify which columns should be one-hot encoded and which should be scaled
categorical_columns = ['type']
numeric_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Create transformers for one-hot encoding and scaling
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False, drop='first'))  # Use drop='first' to prevent multicollinearity
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Apply transformers to columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_columns),
        ('num', numeric_transformer, numeric_columns)
    ])

# Step 3: Preprocess the data and save it as numpy arrays
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
y_train = y_train.values  # Convert y_train to a numpy array

# Step 4: Save the preprocessed data as numpy arrays
np.save("data/X_train.npy", X_train_preprocessed)
np.save("data/X_test.npy", X_test_preprocessed)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test.values)  # Convert y_test to a numpy array
