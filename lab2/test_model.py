# Проврека модели на тестовой выборке (валидация)

# Импорт библиотек
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pickle

# Загрузка тестовой выборки
X_test = pd.read_csv('data/test/X_test.csv')
y_test = pd.read_csv('data/test/y_test.csv')

# Загрузка файла модели
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# ВЫчисление метрик
accuracy = accuracy_score(y_test['target'], y_pred)
precision = precision_score(y_test['target'], y_pred, average='weighted')
recall = recall_score(y_test['target'], y_pred, average='weighted')
f1 = f1_score(y_test['target'], y_pred, average='weighted')

# Создание выходных данных
results = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-score': [f1]
})
# Вывод выходных данных
print('Results on test data:')
print(results.to_string(index=False))