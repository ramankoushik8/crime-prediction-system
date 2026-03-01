import urllib.request
import os

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Direct URL to the UCI Machine Learning Repository data file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
output_path = "data/communities-crime-full.csv"

print(f"Downloading dataset to {output_path}...")
urllib.request.urlretrieve(url, output_path)
print("Download complete and ready for the ETL pipeline!")
