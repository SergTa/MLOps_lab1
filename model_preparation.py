# Создание модели логистической регрессии

# Импорт библиотек
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import os
import sys


# Функция для тренировки модели и вычисления метрик
def train_model_and_evaluate(file_path):
    # Загрузка предобработанных данных
    df = pd.read_csv(file_path)

    # Перемешивание данных
    df = shuffle(df, random_state=42)

    # Разделение данных на признаки и целевую переменную
    X = df[['temperature']]
    y = df['anomaly']

    # Создание модели регрессии
    model = LogisticRegression()

    # Тренировка модели
    model.fit(X, y)

    # Предсказание на тренировочных данных
    y_pred = model.predict(X)

    # ВЫчисление метрик
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Создание Дата-сета из результатов
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return model, results


# Получение номера дата-сета
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # По умолчанию

# Создание директории для сохранения модели
os.makedirs('models', exist_ok=True)
print()
for i in range(n_datasets):
    # Тренировка модели на предобработанных данных
    model, results = train_model_and_evaluate(
        f'train/X_train_{i+1}_prep.csv')

    # Сохранение обученной модели
    joblib.dump(model, f'models/model_{i+1}.pkl')

    print(f"The model on the data set # {i+1} has been tested successfully.")
    print(results.to_string(index=False))



