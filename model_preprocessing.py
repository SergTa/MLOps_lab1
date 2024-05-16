import pandas as pd
import sys

# предварительная обработка числовых признаков
from sklearn.preprocessing import StandardScaler # Импортируем стандартизацию от scikit-learn
from sklearn.preprocessing import PowerTransformer  # Степенное преобразование от scikit-learn
from sklearn.pipeline import Pipeline # Pipeline. Ни добавить, ни убавить

from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок

#загрузка данных
train_df = pd.read_csv(train/data_train.csv)
test_df = pd.read_csv(test/data_test.csv)

class QuantileReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.quantiles = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include='number'):
            low_quantile = X[col].quantile(self.threshold)
            high_quantile = X[col].quantile(1 - self.threshold)
            self.quantiles[col] = (low_quantile, high_quantile)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X.select_dtypes(include='number'):
            low_quantile, high_quantile = self.quantiles[col]
            rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
            if rare_mask.any():
                rare_values = X_copy.loc[rare_mask, col]
                replace_value = np.mean([low_quantile, high_quantile])
                if rare_values.mean() > replace_value:
                    X_copy.loc[rare_mask, col] = high_quantile
                else:
                    X_copy.loc[rare_mask, col] = low_quantile
        return X_copy

num_pipe_distance = Pipeline([
    ('QuantReplace', QuantileReplacer(threshold=0.01, )),
    ('scaler', StandardScaler())
])
num_distance = ['Distance']

num_pipe_engine = Pipeline([
    ('scaler', StandardScaler())
])
num_engine = ['Engine_capacity(cm3)']

num_pipe_year = Pipeline([
    ('power', PowerTransformer())
])
num_year = ['Year']

# Pipeline с числовыми признаками
preprocessors_num = ColumnTransformer(transformers=[
    ('num_distance', num_pipe_distance, num_distance),
    ('num_engine', num_pipe_engine, num_engine),
    ('num_year', num_pipe_year, num_year),
])

# объединяем названия колонок в один список (важен порядок как в ColumnTransformer)
columns_num = np.hstack([num_distance,
                    num_engine,
                    num_year,])

# не забываем удалить целевую переменную цену из признаков
X_train,y_train = train_df.drop(columns = ['Price(euro)']), train_df['Price(euro)']
X_val, y_val = test_df.drop(columns = ['Price(euro)']), test_df['Price(euro)']

# Сначала преобразуем на тренировочных данных
X_train_prep = preprocessors_num.fit_transform(X_train)
# потом на тестовых
X_val_prep = preprocessors_num.transform(X_val)

#Сохраняем скалированные данные
X_train.to_csv(f'/train/X_train.csv', index = False)
y_train.to_csv(f'/train/y_train.csv', index = False)

X_val.to_csv(f'/test/X_val.csv', index = False)
y_val.to_csv(f'/test/y_val.csv', index = False)
