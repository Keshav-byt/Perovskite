import requests
import pandas as pd
import json
import os

# Load dataset
csv_path = "data/perovskite_data.csv"
df = pd.read_csv(csv_path)

# If there's an 'insulator' column, filter to get an insulator sample
if 'insulator' in df.columns:
    insulator_sample = df[df['insulator'] == 1].iloc[0].to_dict()
else:
    raise ValueError("The dataset does not contain an 'insulator' column to filter by.")

# Remove target columns before sending
insulator_sample.pop('pbe band gap', None)
insulator_sample.pop('insulator', None)

# Define API endpoint
url = "http://127.0.0.1:5000/predict"

# Send request
response = requests.post(url, json=insulator_sample)

# Print response
print("Response Status Code:", response.status_code)
print("Response JSON:", json.dumps(response.json(), indent=4))