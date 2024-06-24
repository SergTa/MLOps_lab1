import allure
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


@allure.feature("Translation API")
class TestTranslation:

    @allure.title("Test translation from Russian to English with 404 response")
    @allure.description("""
    Test case to verify the translation from Russian to English with a text that does not exist.

    Steps:
    1. Send a translation request from Russian to English with a non-existent text.
    2. Verify the response status code to be 404.
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_ru_en_404(self):
        with allure.step("Sending translation request from Russian to English"):
            response = client.post("/translate/ru-en/", json={"text": "привет"})
        with allure.step("Verifying response status code"):
            assert response.status_code == 404

    @allure.title("Test translation from Russian to English with 200 response")
    @allure.description("""
    Test case to verify the translation from Russian to English with a valid text.

    Steps:
    1. Send a translation request from Russian to English with a valid text.
    2. Verify the response status code to be 200.
    3. Verify the translated text to be "Hey.".
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_ru_en_ok(self):
        with allure.step("Sending translation request from Russian to English"):
            response = client.post("/translate/ru-to-en/", json={"text": "привет"})
        with allure.step("Verifying response status code and translated text"):
            assert response.status_code == 200
            assert response.json() == {"translated_text": "Hey."}

    @allure.title("Test translation from Russian to English with a text phrase")
    @allure.description("""
    Test case to verify the translation from Russian to English with a text phrase.

    Steps:
    1. Send a translation request with a text phrase from Russian to English.
    2. Verify the response status code to be 200.
    3. Verify the translated text to be "Example of a text phrase.".
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_ru_en_phrase_ok(self):
        with allure.step("Sending translation request with text phrase from Russian to English"):
            response = client.post("/translate/ru-to-en/", json={"text": "Пример текстовой фразы."})
        with allure.step("Verifying response status code and translated text"):
            assert response.status_code == 200
            assert response.json() == {"translated_text": "Example of a text phrase."}

    @allure.title("Test translation from Russian to English with an error")
    @allure.description("""
    Test case to verify the translation from Russian to English with an error in the request.

    Steps:
    1. Send a translation request from Russian to English.
    2. Verify the response status code to be 200.
    3. Verify the translation text not to be "Hi.".
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_ru_en_error(self):
        with allure.step("Sending translation request from Russian to English"):
            response = client.post("/translate/ru-to-en/", json={"text": "привет"})
        with allure.step("Verifying response status code and translation text"):
            assert response.status_code == 200
            assert not response.json() == {"translation_text": "Hi."}

    @allure.title("Test translation from Russian to English with an incorrect request")
    @allure.description("""
    Test case to verify the translation from Russian to English with incorrect request data.

    Steps:
    1. Send a translation request from Russian to English with incorrect data.
    2. Verify the response status code to be 422.
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_ru_en_not_200(self):
        with allure.step("Sending translation request from Russian to English with incorrect data"):
            response = client.post("/translate/ru-to-en/", json={"message": "привет"})
        with allure.step("Verifying response status code"):
            assert response.status_code == 422

    @allure.title("Test translation from English to Russian with 404 response")
    @allure.description("""
    Test case to verify the translation from English to Russian with a text that does not exist.

    Steps:
    1. Send a translation request from English to Russian.
    2. Verify the response status code to be 404.
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_en_ru_404(self):
        with allure.step("Sending translation request from English to Russian"):
            response = client.post("/translate/en-ru/", json={"text": "hello"})
        with allure.step("Verifying response status code"):
            assert response.status_code == 404

    @allure.title("Test translation from English to Russian with 200 response")
    @allure.description("""
    Test case to verify the translation from English to Russian with a valid text.

    Steps:
    1. Send a translation request from English to Russian.
    2. Verify the response status code to be 200.
    3. Verify the translated text to be "Привет.".
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_en_ru_ok(self):
        with allure.step("Sending translation request from English to Russian"):
            response = client.post("/translate/en-to-ru/", json={"text": "hello"})
        with allure.step("Verifying response status code and translated text"):
            assert response.status_code == 200
            assert response.json() == {"translated_text": "Привет."}

    @allure.title("Test translation from English to Russian with a text phrase")
    @allure.description("""
    Test case to verify the translation from English to Russian with a text phrase.

    Steps:
    1. Send a translation request with a text phrase from English to Russian.
    2. Verify the response status code to be 200.
    3. Verify the translated text to be "Пример формулировки текста.".
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_en_ru_phrase_ok(self):
        with allure.step("Sending translation request with text phrase from English to Russian"):
            response = client.post("/translate/en-to-ru/", json={"text": "Example of a text phrase."})
        with allure.step("Verifying response status code and translated text"):
            assert response.status_code == 200
            assert response.json() == {"translated_text": "Пример формулировки текста."}

    @allure.title("Test translation from English to Russian with an error")
    @allure.description("""
    Test case to verify the translation from English to Russian with an error in the request.

    Steps:
    1. Send a translation request from English to Russian.
    2. Verify the response status code to be 200.
    3. Verify the translated text not to be "Не Привет.".
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_en_ru_error(self):
        with allure.step("Sending translation request from English to Russian"):
            response = client.post("/translate/en-to-ru/", json={"text": "hello"})
        with allure.step("Verifying response status code and translated text"):
            assert response.status_code == 200
            assert not response.json() == {"translated_text": "Не Привет."}

    @allure.title("Test translation from English to Russian with an incorrect request")
    @allure.description("""
    Test case to verify the translation from English to Russian with incorrect request data.

    Steps:
    1. Send a translation request from English to Russian with incorrect data.
    2. Verify the response status code to be 422.
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_translate_en_ru_error_not_200(self):
        with allure.step("Sending translation request from English to Russian with incorrect data"):
            response = client.post("/translate/en-to-ru/", json={"message": "hello"})
        with allure.step("Verifying response status code"):
            assert response.status_code == 422
