# ML 


# MLOps 



# IRIS classification
# 🌸 Iris Classification in Machine Learning

## 📘 Overview

The **Iris Classification** problem is a classic example of a **supervised machine learning** task. It uses the **Iris flower dataset**, introduced by **Ronald Fisher** in 1936.

---

## 🌼 The Iris Dataset

The dataset contains **150 samples** from **3 species** of Iris flowers:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

### 📊 Features (Input Variables)

Each flower sample has the following **4 features**:

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

### 🎯 Target (Output Variable)

- A single label indicating the species of the iris flower:
  - 0 → Setosa
  - 1 → Versicolor
  - 2 → Virginica

---

## 🎯 Goal of Iris Classification

The goal is to **build a machine learning model** that can predict the **species** of an iris flower based on its 4 features.

### ✅ Problem Type: 
- **Multiclass Classification**

### ✅ Objective:
- Learn from labeled data to classify new, unseen flower samples accurately.

---

## 🛠️ Steps in Iris Classification

### 1. Load the Dataset

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
````

---

### 2. Explore the Data

```python
import pandas as pd

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df.head()
```

Optional: visualize using seaborn or matplotlib.

---

### 3. Split the Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 4. Train a Classifier (e.g., KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```

---

### 5. Evaluate the Model

```python
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

### 6. Predict New Data

```python
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example feature values
prediction = model.predict(sample)
print("Predicted class:", iris.target_names[prediction][0])
```

---

## 🔍 Why is Iris Classification Popular?

* Small and clean dataset
* Easy to visualize
* Great for understanding classification
* Good for benchmarking simple algorithms

---

## 💡 Common Algorithms Used

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Support Vector Machine (SVM)
* Random Forest
* Naive Bayes

---

## 📌 Summary

| Item       | Description                                          |
| ---------- | ---------------------------------------------------- |
| Problem    | Multiclass Classification                            |
| Dataset    | Iris Flower Dataset                                  |
| Features   | Sepal length, Sepal width, Petal length, Petal width |
| Target     | Iris species (Setosa, Versicolor, Virginica)         |
| Algorithms | KNN, SVM, Decision Tree, etc.                        |
| Goal       | Predict the species from the features                |

---

```

---
```


# MLOps Demo Lab 1
Steps to follow
```
> mkdir mlops-lab1 
> cd mlops-lab1
> python -m venv .venv
> source .venv/bin/activate     # or .venv\Scripts\activate on Windows
> pip install scikit-learn mlflow pandas
```

create a file named train.py
then, write the code as you find in the file. 

then,
execute the commands 
  >python train.py
  
you will see a warning
===
  The accuracy = 1.000 shows your logistic regression model trained perfectly on the Iris dataset (expected for this small dataset).

  The run is already logged in MLflow — you can confirm by opening:

> mlflow ui
and visit http://127.0.0.1:5000

In the app You'll see:

Run entry under iris_classification

Logged parameter: model_type

Logged metric: accuracy

Artifact: model/ (pickled model)


======
## MLflow
MLflow is an open-source tool that helps manage the machine learning lifecycle—tracking experiments, packaging code into reproducible runs, and sharing and deploying models.

On the Experiments page, you see a list of experiments you have created:
here’s an experiment named iris_classification, created on October 10, 2025.

Each experiment entry includes details like when it was created, last modified, a description, and any tags added for organization.

You can create, compare, or delete experiments.

It helps you organize and track different ML experiments—e.g., testing different models or configurations.

You can compare results, keep records, and collaborate efficiently if working on a team.

=========

### Pandas
Pandas is mainly focused on data manipulation, cleaning, and exploration. It’s great for handling data in tabular form and preparing it for analysis.

### Scikit-learn
Scikit-learn, on the other hand, is focused on machine learning. It provides tools and algorithms for training models, making predictions, and evaluating performance.
