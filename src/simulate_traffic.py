import requests
import time
from etl import CrimeDataETL

print("Initializing Data Streamer...")
etl = CrimeDataETL()
# Load the data and extract both features (X) and the actual answers (y)
df = etl.load_data('../data/communities-crime-full.csv')
X, y_actual = etl.transform(df)

# Let's simulate a stream of 20 incoming data reports
num_requests = 20
print(f"Streaming {num_requests} records to the prediction API...\n")
print("-" * 55)
print(f"{'Record ID':<12} | {'Actual Risk':<15} | {'API Prediction':<15}")
print("-" * 55)

for i in range(num_requests):
    # Extract the row and format keys to strings for JSON serialization
    row = X.iloc[i].to_dict()
    features_dict = {str(key): value for key, value in row.items()}
    
    payload = {"features": features_dict}
    actual_val = bool(y_actual.iloc[i])
    
    try:
        # Send the data to your local FastAPI server
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            prediction = response.json().get("highCrime_prediction")
            
            # Print to our terminal dashboard
            match = "✅" if prediction == actual_val else "❌"
            print(f"Dispatch #{i+1:<3} | Actual: {str(actual_val):<7} | Predicted: {str(prediction):<7} {match}")
        else:
            print(f"Dispatch #{i+1:<3} | Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection failed. Is the FastAPI server running?")
        break
        
    # Pause for half a second to simulate live incoming data
    time.sleep(0.5)

print("-" * 55)
print("Live traffic simulation complete!")
