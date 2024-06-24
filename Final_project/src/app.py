from typing import Dict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer


app = FastAPI()

# Загрузка предварительно обученных моделей и токенизаторов для перевода
model_ru_to_en_name = "Helsinki-NLP/opus-mt-ru-en"
tokenizer_ru_to_en = MarianTokenizer.from_pretrained(model_ru_to_en_name)
model_ru_to_en = MarianMTModel.from_pretrained(model_ru_to_en_name)

model_en_to_ru_name = "Helsinki-NLP/opus-mt-en-ru"
tokenizer_en_to_ru = MarianTokenizer.from_pretrained(model_en_to_ru_name)
model_en_to_ru = MarianMTModel.from_pretrained(model_en_to_ru_name)


# Модель запроса для входных данных
class TranslationRequest(BaseModel):
    text: str


class Translator:

    # Функция для перевода текста с русского на английский
    @staticmethod
    def translate_ru_to_en(text: str, max_length: int=100) -> str:
        encoded_text = tokenizer_ru_to_en(text, return_tensors="pt", padding=True, truncation=True)
        translated_text = model_ru_to_en.generate(**encoded_text, max_length=max_length)
        translated_text = tokenizer_ru_to_en.decode(translated_text[0], skip_special_tokens=True)
        return translated_text

    # Функция для перевода текста с английского на русский
    @staticmethod
    def translate_en_to_ru(text: str, max_length: int=100) -> str:
        encoded_text = tokenizer_en_to_ru(text, return_tensors="pt", padding=True, truncation=True)
        translated_text = model_en_to_ru.generate(**encoded_text, max_length=max_length)
        translated_text = tokenizer_en_to_ru.decode(translated_text[0], skip_special_tokens=True)
        return translated_text


# Эндпоинт для перевода с русского на английский
@app.post("/translate/ru-to-en/")
def translate_ru_to_en_api(request: TranslationRequest) -> Dict:
    translated_text = Translator.translate_ru_to_en(request.text)
    return {"translated_text": translated_text}


# Эндпоинт для перевода с английского на русский
@app.post("/translate/en-to-ru/")
def translate_en_to_ru_api(request: TranslationRequest) -> Dict:
    translated_text = Translator.translate_en_to_ru(request.text)
    return {"translated_text": translated_text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
