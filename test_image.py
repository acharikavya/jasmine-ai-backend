import requests

url = "http://localhost:8000/predict"
files = {"image": open("D:/jamine_ai/sample_images/1.21.jpeg", "rb")}
r = requests.post(url, files=files)
print("Status:", r.status_code)
print(r.json())