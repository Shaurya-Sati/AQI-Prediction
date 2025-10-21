#predict module
model = joblib.load("aqi_model.pkl")
scaler = joblib.load("scaler.pkl")

pm25 = float(input("Enter PM2.5 value: "))
pm10 = float(input("Enter PM10 value: "))
no2 = float(input("Enter NO2 value: "))
so2 = float(input("Enter SO2 value: "))
co = float(input("Enter CO value: "))
o3 = float(input("Enter O3 value: "))

new_data = [[pm25, pm10, no2, so2, co, o3]]
new_data_scaled = scaler.transform(new_data)
predicted_aqi = model.predict(new_data_scaled)[0]
print(f"Predicted AQI: {round(predicted_aqi, 2)}")

if predicted_aqi <= 50:
    category = "Good"
elif predicted_aqi <= 100:
    category = "Satisfactory"
elif predicted_aqi <= 200:
    category = "Moderate"
elif predicted_aqi <= 300:
    category = "Poor"
elif predicted_aqi <= 400:
    category = "Very Poor"
else:
    category = "Severe"
print("Air Quality Category:", category)

pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
values = [pm25, pm10, no2, so2, co, o3]
plt.bar(pollutants, values, color='skyblue', edgecolor='black')
plt.xlabel('Pollutants')
plt.ylabel('Concentration (µg/m³)')
plt.title(f'Pollutant Levels (Predicted AQI: {round(predicted_aqi, 2)} | {category})')
plt.show()