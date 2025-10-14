import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set up MLflow experiment
mlflow.set_experiment("iris_classification")

# Load the iris dataset and split into train/test sets
iris = load_iris()
# print(iris) # loaded dataset
# print(iris.data)  # features
print("**************************")
# print(iris.target) # target labels

# Let's split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,  # features
    iris.target,  # targets
    test_size=0.2, # 20% test data
    random_state=42 # for reproducibility
)

# print("X_train:", X_train)
# print("y_train:", y_train)
# print("X_test:", X_test)
# print("y_test:", y_test)

# # Start MLflow run
with mlflow.start_run():
    # Train model
    model = LogisticRegression(max_iter=200)
    # Fit the model 
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    preds = model.predict(X_test) # remaining 30 samples 
    # preds are the predicted targets (0 / 1 / 2) for the test set
    
    # comparing predictions vs actual labels to find out accuracy
    acc = accuracy_score(y_test, preds) 

    # Log to MLflow
    mlflow.log_param("model_type", "logistic_regression") # Log which model we are using
    mlflow.log_metric("accuracy", acc)  # Log the accuracy value of the model
    mlflow.sklearn.log_model(model, "model") # Save the trained model as an artifact

    print(f"Run logged to MLflow. \n Accuracy: {acc:.3f}")
