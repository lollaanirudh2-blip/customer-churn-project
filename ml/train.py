import pandas as pd

# Load data
df = pd.read_csv("data/dataset.csv")

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing
df = df.dropna()

# Drop ID
df = df.drop('customerID', axis=1)

# Convert target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical
df = pd.get_dummies(df, drop_first=True)

print("Cleaned Data Shape:", df.shape)
print(df.head())