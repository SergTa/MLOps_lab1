A simple automated machine learning system
This is an implementation of a simple automated machine learning system using Docker and Streamlit to deploy a logistic regression model on the Iris dataset.

How it works
The work process includes the following steps.

1. Training of the model
Based on the Iris dataset, using the sklearn library, a logistic regression model is created.

This process includes:

uploading a dataset,
dividing it into training and test sets,
normalisation,
model training on a training set,
saving the trained model and scaler for later use.
2. Containerisation with Docker
A separate Dockerfile is created for each part of the application (model and web interface), which defines how to create a Docker image for each part.

This includes installing all necessary dependencies and copying the application code into the image.

Docker Compose is used to manage the startup and communication between the model and web interface containers.

3. Web interface with Streamlit
`Streamlit' is used to create an interactive web interface that allows users to enter parameters to predict the Iris species.

The user can use the sliders to enter sepal and petal length and width, and then click the "Predict" button to get the prediction from the model.

4. Integration of model and web interface
The model and web interface work together to allow users to input data through the web interface and receive predictions from the model.

The input data vector received from the user is normalised using a stored scaler and then passed to the model to produce a prediction.

The prediction is then displayed to the user in the web interface.

5. Deployment and management
Using Docker and Docker Compose makes it easier to deploy and manage the application.

`Docker Compose' makes it easy to run and manage multiple containers as a single system, providing uniformity and isolation between different parts of the application. This simplifies the process of deploying and updating the application, and provides easy scalability and version control.

How to get started
Clone the repository.
Go to the project directory:
cd simple_automatic_machine_learning_pipeline/lab3
Run Docker Compose:
docker-compose up
Open http://localhost:8501 in your browser to access the Streamlit application.
How to use
The Streamlit app allows you to enter parameters to predict Iris species.

Use the sliders to enter the length and width of the sepal and petal, and then click "Predict" to get the model prediction.

Application screenshot

Лицензия
This project is distributed under the MIT licence.