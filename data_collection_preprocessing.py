# ----------------------------------------------------------
# MODULE 1: DATA COLLECTION & PREPROCESSING
# Project: Air Quality Index Prediction using Python
# Team: Avengers
# ----------------------------------------------------------

# Step 1: Import the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset
# Make sure air_quality_dataset.csv is in the same folder as this code
df = pd.read_csv("air_quality_dataset.csv")

# Step 3: Display first few rows to understand the data
print("🔹 Sample data:\n", df.head())

# Step 4: Check for missing or null values
print("\n🔹 Missing values in each column:")
print(df.isnull().sum())

# Step 5: Fill missing values (if any) with mean
df = df.fillna(df.mean(numeric_only=True))

print("\n✅ After handling missing values:")
print(df.head())

# Step 6: Select input features (pollutants) and output target (AQI)
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
target = 'AQI'

X = df[features]   # independent variables (inputs)
y = df[target]     # dependent variable (output)

# Step 7: Normalize the feature data (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Feature scaling completed.")
print("Scaled Data (first 5 rows):\n", X_scaled[:5])

# Step 8: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n✅ Data successfully split into train and test sets.")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 9: Display confirmation message
print("\n🎯 MODULE 1 COMPLETED SUCCESSFULLY")
