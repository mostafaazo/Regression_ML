import requests

url = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint
body = {
    "est_num1": 2.2,
    "est_num2": 2.3,
    "est_letter": "j",
}
response = requests.post(url, data=body)
response.json()