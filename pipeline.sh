#!/bin/bash

# Из-за того, что scikit-learn невозможно установить на LINUX без создания виртуального окружения,
# создадим виртуальное окружение, активируем его, установим зависимости из файла requirements.txt

create_venv() {
    local env_name=${1:-".venv"}
    python3 -m venv "$env_name"
    echo "The virtual environment '$env_name' has been created."
}

# Function to activate the virtual environment
activate_venv() {
    local env_name=${1:-".venv"}
    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create."
        return 1
    fi
    if [ -z "$VIRTUAL_ENV" ]; then
        source "./$env_name/bin/activate"
        echo "Virtual environment '$env_name' is activated."
    else
        echo "The virtual environment has already been activated."
    fi
}

# Function for installing dependencies from requirements.txt
install_deps() {
    if [ ! -f "requirements.txt" ]; then
        echo "File requirements.txt not found."
        return 1
    fi

    # Check if all dependencies from requirements.txt are installed
    for package in $(cat requirements.txt | cut -d '=' -f 1); do
        if ! pip freeze | grep -q "^$package=="; then
            echo "Dependency installation..."
            pip install -r requirements.txt
            echo "Dependencies installed."
            return 0
        fi
    done

    echo "All dependencies are already installed."
}

# Creating a virtual environment, if not already created
if [ ! -d ".venv" ]; then
    create_venv
fi

# Activating the virtual environment
activate_venv

# Dependency installation
install_deps



n_datasets=$1 # Установка номера дата-сета

python3 /home/serg/MLOps/Lab_1/MLOps_lab1/data_creation.py $n_datasets
python3 /home/serg/MLOps/Lab_1/MLOps_lab1/model_preprocessing.py $n_datasets
python3 /home/serg/MLOps/Lab_1/MLOps_lab1/model_prepfrftion.py $n_datasets
python3 /home/serg/MLOps/Lab_1/MLOps_lab1/model_testing.py $n_datasets
