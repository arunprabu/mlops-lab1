# Import necessary libraries for MLflow experiment tracking and machine learning

import mlflow  # MLflow library for experiment tracking and model management
# MLflow is primarily focused on managing the machine learning lifecycle, 
# including tracking experiments, packaging code, and deploying models.

import mlflow.sklearn  # MLflow integration for scikit-learn models
from sklearn.datasets import load_iris  # Built-in iris dataset for classification
from sklearn.model_selection import train_test_split  # Function to split data into train/test sets
from sklearn.linear_model import LogisticRegression  # Logistic regression classifier
from sklearn.metrics import accuracy_score  # Metric to evaluate model performance
import numpy as np  # NumPy for numerical operations

# Set up MLflow experiment for organizing related runs
# This creates or uses an existing experiment named "iris_classification"
mlflow.set_experiment("iris_classification")

# Load the iris dataset and split into training and testing sets
data = load_iris()  # Load the iris flower dataset (150 samples, 4 features, 3 classes)
X_train, X_test, y_train, y_test = train_test_split(
    data.data,      # Feature matrix (sepal/petal length and width)
    data.target,    # Target labels (0: setosa, 1: versicolor, 2: virginica)
    test_size=0.2,  # Use 20% of data for testing, 80% for training
    random_state=42 # Set random seed for reproducible results
)

# Start an MLflow run to track this training experiment
with mlflow.start_run():
    # Create and train a logistic regression model
    model = LogisticRegression(max_iter=200)  # Set max iterations to ensure convergence
    model.fit(X_train, y_train)  # Train the model on training data
    
    # Make predictions on the test set and calculate accuracy
    preds = model.predict(X_test)  # Generate predictions for test data
    acc = accuracy_score(y_test, preds)  # Calculate accuracy score (correct predictions / total predictions)

    # Log experiment parameters and metrics to MLflow for tracking
    mlflow.log_param("model_type", "logistic_regression")  # Log the type of model used
    mlflow.log_metric("accuracy", acc)  # Log the accuracy metric for this run

    input_example = np.array([X_test[0]])
    mlflow.sklearn.log_model(model, "model", input_example=input_example) # Save the trained model artifact

    # Print results to console
    print(f"Run logged to MLflow. Accuracy: {acc:.3f}")


