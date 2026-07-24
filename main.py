import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

print("Credit Card Fraud Detection System")
print("----------------------------------")

# Take input from user
merchant = int(input("Merchant: "))
category = int(input("Category: "))
amt = float(input("Transaction Amount: "))
gender = int(input("Gender (0=Female, 1=Male): "))
city = int(input("City: "))
state = int(input("State: "))
zip_code = int(input("ZIP Code: "))
lat = float(input("Latitude: "))
long = float(input("Longitude: "))
city_pop = int(input("City Population: "))
job = int(input("Job: "))
unix_time = int(input("Unix Time: "))
merch_lat = float(input("Merchant Latitude: "))
merch_long = float(input("Merchant Longitude: "))

# Create input array
data = np.array([[merchant, category, amt, gender, city,
                  state, zip_code, lat, long, city_pop,
                  job, unix_time, merch_lat, merch_long]])

# Prediction
prediction = model.predict(data)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("⚠ Fraud Transaction")
else:
    print("✓ Genuine Transaction")