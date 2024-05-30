# Python image
FROM python:3.8

# Setting the working directory in the container
WORKDIR /models

# Copying files
COPY . .

# Dependency installation from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Launching the application
CMD ["python3", "models_py.py"]