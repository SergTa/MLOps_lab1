import pandas as pd # Библиотека Pandas для работы с табличными данными
import os
from sklearn.model_selection import train_test_split#  функция разбиения на тренировочную и тестовую выборку

#Создание папок для тренировочной и тестовой выборки
os.makedirs ('train', exist_ok = True)
os.makedirs ('test', exist_ok = True)

#Набор данных
#Набор данных представляет собой статистику параметров автомобилей на вторичном рынке в Молдавии.
#Набор включает ряд категориальных и численных значений, составляющих одну запись (строку). Число записей можно найти как число строк.
#Каждый столбец в записи — это отдельный параметр.
#Среди указанных параметров приведен целевой для задачи предсказания (регрессии) - цена автомобиля.

DF = pd.read_csv('https://raw.githubusercontent.com/dayekb/mpti_ml/main/data/cars_moldova_clean.csv', delimiter = ',')

# не забываем выделить целевую переменную цену из признаков
X,y = DF.drop(columns = ['Price(euro)']), DF['Price(euro)']

# разбиваем на тестовую и тренировочную выборку 
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

#Сохраняем данные
df_train = pd.DataFrame(X_train, y_train)
df_train.to_csv(f'train/data_train.csv', index = False)

df_test = pd.DataFrame(X_val, y_val)
df_test.to_csv(f'test/data_test.csv', index = False)
