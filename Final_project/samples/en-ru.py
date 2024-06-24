import requests

url = "http://91.185.85.43:8000/translate/en-to-ru/"
payload = {"text": "If you want to be OK, drinking vodka every day!"}
response = requests.post(url, json=payload)
print(response.json())
