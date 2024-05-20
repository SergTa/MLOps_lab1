#!/bin/bash

n_datasets=$1 # Установка номера дата-сета

python3 /home/serg/MLOps/Lab_1/MLOps_lab1/data_creation.py $n_datasets
python3 /home/serg/MLOps/Lab_1/MLOps_lab1/model_preprocessing.py $n_datasets
python3 /home/serg/MLOps/Lab_1/MLOps_lab1/model_preparation.py $n_datasets
python3 /home/serg/MLOps/Lab_1/MLOps_lab1/model_testing.py $n_datasets
