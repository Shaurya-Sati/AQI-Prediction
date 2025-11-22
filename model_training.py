import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("aqi_data.csv")

X = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
y = data['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

joblib.dump(model, "aqi_model.pkl")
joblib.dump(scaler, "aqi_scaler.pkl")

accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
