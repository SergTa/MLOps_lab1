import requests

url = "http://91.185.85.43:8000/translate/ru-to-en/"
payload = {"text": "Если хочешь остаться, останься просто так"}
response = requests.post(url, json=payload)
print(response.json())
