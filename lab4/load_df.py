#Загрузка библиотек
from catboost.datasets import titanic
import pandas as pd


# Загрузка датасета
df, _ = titanic()

# Сохранение
df.to_csv('./datasets/titanic_df.csv', index=False)