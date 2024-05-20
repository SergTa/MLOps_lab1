#!/bin/bash

n_datasets=$1 # Установка номера дата-сета
python data_creation.py 
python model_preprocessing.py
python model_prepfrftion.py
python model_testing.py
