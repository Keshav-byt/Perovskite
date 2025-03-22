import requests

API_URL = "http://127.0.0.1:5000/predict"

input_data = {
    "functional group": "AgBaMoCdO6",
    "A": "Ag",
    "A_OS": 1.0,
    "A'": "Ba",
    "A'_OS": 1.5,
    "A_HOMO-": -5.2,
    "A_HOMO+": -3.1,
    "A_IE-": 6.5,
    "A_IE+": 7.2,
    "A_LUMO-": -2.1,
    "A_LUMO+": -1.3,
    "A_X-": 3.1,
    "A_X+": 3.5,
    "A_Z_radii-": 1.2,
    "A_Z_radii+": 1.4,
    "A_e_affin-": 2.5,
    "A_e_affin+": 3.0,
    "Bi": "Mo",
    "B_OS": 2.0,
    "B'": "Cd   ",
    "B'_OS": 2.5,
    "B_HOMO-": -4.8,
    "B_HOMO+": -2.9,
    "B_IE-": 5.8,
    "B_IE+": 6.3,
    "B_LUMO-": -1.9,
    "B_LUMO+": -1.1,
    "B_X-": 2.8,
    "B_X+": 3.2,
    "B_Z_radii-": 1.1,
    "B_Z_radii+": 1.3,
    "B_e_affin-": 2.2,
    "B_e_affin+": 2.7,
    "μ": 3.5,
    "μĀ": 4.1,
    "μ𝐵̅": 2.9,
    "t": 0.9
}

# Send request
response = requests.post(API_URL, json=input_data)

# Print response
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)
