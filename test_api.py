import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = {
    "history": [
        {"voltage": 230, "current": 11.2, "temperature": 42, "vibration": 1.1, "speed": 1485, "load_percent": 85},
        {"voltage": 230, "current": 11.2, "temperature": 42, "vibration": 1.1, "speed": 1485, "load_percent": 85},
        {"voltage": 230, "current": 11.2, "temperature": 42, "vibration": 1.1, "speed": 1485, "load_percent": 85},
        {"voltage": 230, "current": 11.2, "temperature": 42, "vibration": 1.1, "speed": 1485, "load_percent": 85},
        {"voltage": 230, "current": 11.2, "temperature": 42, "vibration": 1.1, "speed": 1485, "load_percent": 85},
        {"voltage": 195, "current": 19.0, "temperature": 110, "vibration": 8.5, "speed": 1370, "load_percent": 125}
    ]
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error connecting to API: {e}")
