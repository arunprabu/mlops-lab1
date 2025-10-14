# Import necessary libraries for MLflow experiment tracking and machine learning
import mlflow  # MLflow is a tool to track experiments, manage models, and deploy them
import mlflow.sklearn  # For tracking scikit-learn models specifically
from sklearn.datasets import load_iris  # We will use the Iris dataset (a popular dataset for classification)
from sklearn.model_selection import train_test_split  # Splits data into training and testing sets
from sklearn.linear_model import LogisticRegression  # A simple algorithm for classification (logistic regression)
from sklearn.metrics import accuracy_score  # Metric to evaluate how well the model performs

# Step 1: Set up MLflow experiment to organize related runs
mlflow.set_experiment("iris_classification")  # Create or use an experiment called 'iris_classification'

# Step 2: Load the Iris dataset and split it into training and test data
data = load_iris()  # Load the Iris dataset (features: sepal/petal length/width, target: species)
X_train, X_test, y_train, y_test = train_test_split(
    data.data,      # Features: sepal and petal length/width
    data.target,    # Target labels: species (0: setosa, 1: versicolor, 2: virginica)
    test_size=0.2,  # 20% of the data will be used for testing, the rest for training
    random_state=42 # Ensures that the data split is reproducible (same every time)
)

# Step 3: Start an MLflow run to track the experiment
with mlflow.start_run():
    # Step 4: Create a logistic regression model and train it
    model = LogisticRegression(max_iter=200)  # Create a logistic regression model (algorithm)
    model.fit(X_train, y_train)  # Train the model using the training data
    
    # Step 5: Make predictions using the test data
    preds = model.predict(X_test)  # Predict which species the model thinks the test data belongs to
    
    # Step 6: Calculate the accuracy of the model (how many predictions were correct)
    acc = accuracy_score(y_test, preds)  # Compare model predictions to actual labels and calculate accuracy

    # Step 7: Log important information about this experiment to MLflow
    mlflow.log_param("model_type", "logistic_regression")  # Log which model we are using
    mlflow.log_metric("accuracy", acc)  # Log the accuracy value of the model
    
    # Step 8: Log the trained model (saving it for later use)
    mlflow.sklearn.log_model(model, "model")  # Save the trained model as an artifact

    # Print the accuracy to the console
    print(f"Run logged to MLflow. Accuracy: {acc:.3f}")
