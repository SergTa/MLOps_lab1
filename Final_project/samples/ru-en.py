import requests

url = "http://91.185.85.43:8000/translate/ru-to-en/"
payload = {"text": "Быстрая коричневая лиса перепрыгивает через ленивую собаку."}
response = requests.post(url, json=payload)
print(response.json())
