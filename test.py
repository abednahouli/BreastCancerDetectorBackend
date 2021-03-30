import requests

BASE = "http://127.0.0.1:5000/"

response = requests.post(BASE + "api/v1/model/predict",{"likes": 3})
print("onApiRequest:"+response.json)