# Приложение "Translator"  

Переводит текст на русский и английский языки с помощью API переводчика.

## Требования
Для развертывания и запуска приложения необходимо иметь установленные следующие компоненты:
- Docker
- Python 3
- pip
- Jenkins
- Allure

## Настройка пайплайна Jenkins:

Установите и настройте Jenkins на сервере.
Создайте новый проект и укажите в качестве источника кода репозиторий проекта.
Скопируйте содержимое файла Jenkinsfile в ваш пайплайн.

### Запуск пайплайна:
Запустите пайплайн в Jenkins и наблюдайте за его выполнением через веб-интерфейс.
Пайплайн можно запустить с кастомной ветки, указав ее в параметрах запуска.

### Последовательность действий в пайплайне
Остановка и удаление старого контейнера и образа: Предварительно останавливает и удаляет предыдущий контейнер и образ приложения, если они существуют.
Подготовка: Очищает рабочее пространство и проверяет исходный код из репозитория.
Проверка: Проверяет указанную ветку из Git-репозитория.
Загрузка набора данных: Загружает необходимые наборы данных с использованием Data Version Control (DVC).
Сборка образа: Создает Docker-образ с именем 'translator-img' для приложения переводчика.
Запуск контейнера: Запускает контейнер Docker с именем 'translator-app' на основе созданного образа.
Установка зависимостей: Настраивает виртуальное окружение, устанавливает зависимости Python, перечисленные в requirements.txt.
Запуск тестов: Выполняет автоматизированные тесты с использованием pytest и генерирует отчеты о тестах Allure.
Генерация отчета Allure: После выполнения тестов формируется отчет Allure, содержащий результаты тестирования.

## Использование API
После запуска приложения вы можете использовать следующие эндпоинты для перевода текста:

POST /translate/ru-to-en/: Перевод текста с русского на английский.
POST /translate/en-to-ru/: Перевод текста с английского на русский.

## Пример запроса:

json { "text": "Привет, мир!" }

Ответ:

json { "translated_text": "Hello, world!" }