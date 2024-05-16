import pandas as pd # Библиотека Pandas для работы с табличными данными

#Набор данных
#Набор данных представляет собой статистику параметров автомобилей на вторичном рынке в Молдавии.
#Набор включает ряд категориальных и численных значений, составляющих одну запись (строку). Число записей можно найти как число строк.
#Каждый столбец в записи — это отдельный параметр.
#Среди указанных параметров приведен целевой для задачи предсказания (регрессии) - цена автомобиля.

DF = pd.read_csv('https://raw.githubusercontent.com/dayekb/mpti_ml/main/data/cars_moldova_clean.csv', delimiter = ',')

#Выбор числовых признаков из загруженного дата-сета
num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']
