import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Sample perovskite data - simplified
test_data = {
    "functional group": "Ba2HfSiO6",
    "A": "Ba",
    "A'": "Ba",
    "Bi": "Hf",
    "B'": "Si",
    "A_IE+": 502.9,
    "B_IE+": 730.95
}

# Make POST request to API
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(test_data), headers=headers)

# Print response
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=4))