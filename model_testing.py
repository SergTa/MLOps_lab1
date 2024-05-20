# Тестирование модели (валидация)

# Импорт библиотек
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import sys


# Функция для теста модели
def test_model(model_path, test_data_path):
    # Загрузка модели
    model = joblib.load(model_path)

    # Загрузка тестовых данных
    df_test = pd.read_csv(test_data_path)

    # Отделение признаков и целевой переменной
    X_test = df_test[['temperature']]
    y_test = df_test['anomaly']

    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Вычисление метрик
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Создание дата-сета с результатами
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return results


# Получение номера дата-сета
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # По умолчанию
print()
for i in range(n_datasets):
    # Путь к модели
    model_path = f'models/model_{i+1}.pkl'
    # Путь к тестовым данным
    test_data_path = f'test/X_test_{i+1}_prep.csv'

    # Тестирование модели
    results = test_model(model_path, test_data_path)
    print(f"The model on the dataset # {i+1} has been tested successfully.")
    print(results.to_string(index=False))
    print('-' * 20)