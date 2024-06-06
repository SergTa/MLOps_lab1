#Загрузка библиотек
import pandas as pd


# Загрузка датасета
df = pd.read_csv('./datasets/titanic_df.csv')

# Применение one-hot-encoding для полового признака
one_hot_enc = pd.get_dummies(df['Sex'], prefix='Sex')

# Добавление новых признаков в исходный DataFrame
df = pd.concat([df, one_hot_enc], axis=1)

# Сохранение новой версии датасета
df.to_csv('./datasets/titanic_df.csv', index=False)
