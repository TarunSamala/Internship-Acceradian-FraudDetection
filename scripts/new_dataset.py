import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Generate synthetic feature data for X_new
num_samples = 60000  # Adjust this number as needed

# Create random data for the features
step = np.random.randint(1, 745, num_samples)
transaction_type = np.random.choice(["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"], num_samples)
amount = np.random.uniform(1, 10000, num_samples)
nameOrig = np.random.choice(["customer_A", "customer_B", "customer_C", "customer_D", "customer_E"], num_samples)
oldbalanceOrg = np.random.uniform(0, 5000, num_samples)
newbalanceOrig = oldbalanceOrg - amount
nameDest = np.random.choice(["recipient_A", "recipient_B", "recipient_C", "recipient_D", "recipient_E"], num_samples)
oldbalanceDest = np.random.uniform(0, 5000, num_samples)
newbalanceDest = oldbalanceDest + amount
isFraud = np.random.choice([0, 1], num_samples)
isFlaggedFraud = np.random.choice([0, 1], num_samples)

# Create a DataFrame for X_new
columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 
           'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']
X_new_df = pd.DataFrame({'step': step, 'type': transaction_type, 'amount': amount,
                         'nameOrig': nameOrig, 'oldbalanceOrg': oldbalanceOrg,
                         'newbalanceOrig': newbalanceOrig, 'nameDest': nameDest,
                         'oldbalanceDest': oldbalanceDest, 'newbalanceDest': newbalanceDest,
                         'isFraud': isFraud, 'isFlaggedFraud': isFlaggedFraud})

# Define categorical and numeric columns
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

# Preprocess the data
X_new_preprocessed = preprocessor.fit_transform(X_new_df)

# Save X_new_preprocessed and y_new as .npy files
np.save("data/X_new.npy", X_new_preprocessed)
np.save("data/y_new.npy", isFraud)
