import pandas as pd
import numpy as np # библиотека Numpy для операций линейной алгебры и прочего
import matplotlib.pyplot as plt # библиотека Matplotlib для визуализации
import seaborn as sns # библиотека seaborn для визуализации
import joblib 


from sklearn.model_selection import ShuffleSplit # при кросс-валидации случайно перемешиваем данные
from sklearn.model_selection import cross_validate # функция кросс-валидации от Scikit-learn

from sklearn.metrics import mean_squared_error as mse # метрика MSE от Scikit-learn
from sklearn.metrics import r2_score # коэффициент детерминации  от Scikit-learn

from sklearn.metrics import PredictionErrorDisplay # Класс визуализации ошибок модели

model = joblib.load('models/model.pkl')
X_train = pd.read_csv(train/X_train.csv)
y_train = pd.read_csv(train/y_train.csv)
X_val = pd.read_csv(test/X_val.csv)
y_val = pd.read_csv(test/y_val.csv)

#Оценка метрик
def calculate_metric(model_pipe, X, y, metric = r2_score):
    """Расчет метрики.
    Параметры:
    ===========
    model_pipe: модель или pipeline
    X: признаки
    y: истинные значения
    metric: метрика (r2 - по умолчанию)
    """
    y_model = model_pipe.predict(X)
    return metric(y, y_model)

print(f"r2 на тренировочной выборке: {calculate_metric(model, X_train, y_train):.4f}")
print(f"r2 на валидационной выборке: {calculate_metric(model, X_val, y_val):.4f}")

print(f"mse на тренировочной выборке: {calculate_metric(model, X_train, y_train, mse):.4f}")
print(f"mse на валидационной выборке: {calculate_metric(model, X_val, y_val, mse):.4f}")
