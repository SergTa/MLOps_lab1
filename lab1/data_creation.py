# Синтез и сохранение в созданных директориях тренировочного и тестового дата-сета температуры
# Внесение аномалий нормальным законом распределения (белый шум)

import numpy as np
import pandas as pd
import os
import sys

# Создание директорий тренировочной и тестовой 'train' и 'test'
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)


# Синтез данных
def generate_data(n_samples,
                  anomaly_ratio=0.1,
                  anomaly_loc=30,
                  anomaly_scale=10):

    # Синтез данных без аномалий
    data = np.random.normal(loc=20, scale=5, size=(n_samples, 1))

    # ВЫчисление числа аномалий
    n_anomalies = int(n_samples * anomaly_ratio)

    # Добавление аномалий
    anomalies = np.random.normal(loc=anomaly_loc, scale=anomaly_scale,
                                 size=(n_anomalies, 1))
    data = np.concatenate((data, anomalies), axis=0)

    # Округление до одного знака после запятой
    data = np.round(data, 1)

    # Синтез второй колонки с указателями аномалии
    labels = np.zeros(data.size, dtype=int)
    labels[n_samples:] = 1  # Цифра 1 указывает на аномалию

    # Создание структурированного массива (списка кортежей)
    dtype = [('data', np.float32), ('labels', np.int32)]
    data_with_labels = np.empty(data.size, dtype=dtype)
    data_with_labels['data'] = data.flatten()
    data_with_labels['labels'] = labels

    # Создание словаря из списка кортежей
    data_dict = {'temperature': [temp for temp, anomaly in data_with_labels],
                 'anomaly': [anomaly for temp, anomaly in data_with_labels]}

    return data_dict


# Получение номера дата-сета
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Значение по умолчанию, если аргумент пропущен

for i in range(n_datasets):
    # Синтез и сохраненине тренировочных данных
    train_data = generate_data(1000,
                               anomaly_ratio=0.1,
                               anomaly_loc=30+i*5,
                               anomaly_scale=10+i*2)
    df_train = pd.DataFrame(train_data)
    df_train.to_csv(f'train/X_train_{i+1}.csv', index=False)

    # Синтез и сохранение тестовых данных
    test_data = generate_data(200,
                              anomaly_ratio=0.1,
                              anomaly_loc=30+i*5,
                              anomaly_scale=10+i*2)
    df_test = pd.DataFrame(test_data)
    df_test.to_csv(f'test/X_test_{i+1}.csv', index=False)