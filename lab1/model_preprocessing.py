# Предобработка загружаемых данных с помощью StandardScaler,
# Сохранение предобработанных данных

# Импорт библиотек
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler


# Создание функции для предобработки данных
def preprocess_data(train_file_path, test_file_path):
    # Train data loading
    train_df = pd.read_csv(train_file_path)
    # Test data loading
    test_df = pd.read_csv(test_file_path)

    # Создание предобработчика  StandardScaler
    scaler = StandardScaler()

    # Обучение предобработчиком StandardScaler тренировочных данных
    scaler.fit(train_df[['temperature']])

    # Применение StandardScaler к тренировочным данным
    train_scaled_data = scaler.transform(train_df[['temperature']])
    # Применение StandardScaler к тестовым
    test_scaled_data = scaler.transform(test_df[['temperature']])

    # Сохранение скалированных данных
    train_df['temperature'] = train_scaled_data
    train_df.to_csv(
        train_file_path.replace('.csv', '_prep.csv'), index=False)

    # Saving scaled test data
    test_df['temperature'] = test_scaled_data
    test_df.to_csv(
        test_file_path.replace('.csv', '_prep.csv'), index=False)


# Получение номера дата-сета
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Значение по умолчанию

for i in range(n_datasets):
    # Предобработка и сохранение данных
    preprocess_data(
        f'train/X_train_{i+1}.csv',
        f'test/X_test_{i+1}.csv')