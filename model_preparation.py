import pandas as pd
from sklearn.neighbors import KNeighborsRegressor # Регрессия К-Ближайших соседей от scikit-learn


import os
import sys
from joblib import dump, load

X_train = pd.read_csv('train/X_train.csv')
y_train = pd.read_csv('train/y_train.csv')
X_val = pd.read_csv('test/X_val.csv')
y_val = pd.read_csv('test/y_val.csv')

#Обучаем модель
model = KNeighborsRegressor(n_neighbors = 3,
                            weights = 'distance')

model.fit(X_train, y_train)


dump(model, 'model.joblib')  # чтобы сохранить объект


