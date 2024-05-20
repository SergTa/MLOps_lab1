#Загрузка данных из sklearn.datasets, выделение признаков и целевой переменной
# Трансформация в DataFrame
# Создание папок и сохранение полученного DataFrame

import pandas as pd
import os
from sklearn.datasets import load_wine

# Data loading
wine = load_wine()
X = wine.data
y = wine.target

# Transforming data in DataFrame
df = pd.DataFrame(data=X, columns=wine.feature_names)
df['target'] = y

print(df.info())
print(df.describe())

# Создание директории
os.makedirs('data', exist_ok=True)

# Сохранение данных в CSV формате
df.to_csv('data/wine.csv', index=False)