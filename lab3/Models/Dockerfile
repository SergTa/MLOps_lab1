# Python image
FROM python:3.8-slim

# Setting the working directory in the container
WORKDIR /Models

# Copying files
COPY . .

# Dependency installation from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Launching the application
CMD ["python", "models_py.py"]