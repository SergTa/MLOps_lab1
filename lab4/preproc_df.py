#Загрузка библиотек
import pandas as pd

# Загрузка датасета
df = pd.read_csv('../datasets/titanic_df.csv')

# Заполнение пропущенных значений в поле "Age" средним значением
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

# Сохранение новой версии датасета
df.to_csv('../datasets/titanic_df.csv', index=False)