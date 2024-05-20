#!/bin/python

n_datasets=$1 # Установка номера дата-сета

python data_creation.py $n_datasets
python model_preprocessing.py $n_datasets
python model_prepfrftion.py $n_datasets
python model_testing.py $n_datasets
