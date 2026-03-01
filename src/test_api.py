import requests
from etl import CrimeDataETL

print("1. Loading a real record from the dataset...")
etl = CrimeDataETL()
df = etl.load_data('../data/communities-crime-full.csv')
X, y = etl.transform(df)

# Grab the very first row of data and convert it to a dictionary
first_row = X.iloc[0].to_dict()

# JSON requires keys to be strings, so we convert the integer column names to strings
features_dict = {str(key): value for key, value in first_row.items()}

# This matches the exact Pydantic schema we built in app.py
payload = {
    "features": features_dict
}

print("2. Sending the record to the local API...")
try:
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    
    print("\n--- API RESPONSE ---")
    print(f"Status Code: {response.status_code}")
    print(f"Prediction Body: {response.json()}")
    print("--------------------\n")
    
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API. Is your Uvicorn server running?")
