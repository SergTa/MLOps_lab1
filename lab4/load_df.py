#Загрузка библиотек
from catboost.datasets import titanic
import pandas as pd


# Загрузка датасета
df, _ = titanic()

# Сохранение в CSV
df.to_csv('../datasets/titanic/titanic_df.csv', index=False)