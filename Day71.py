import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load Breast Cancer Dataset
cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

print("Dataset Loaded Successfully")
print("Dataset Shape:", X.shape)

print("\nFirst 5 Records:")
print(X.head())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Default KNN Model
default_knn = KNeighborsClassifier()

default_knn.fit(X_train, y_train)

default_predictions = default_knn.predict(X_test)

# Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Define Hyperparameter Grid
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 13],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

# Apply GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train using Training Data
grid_search.fit(X_train, y_train)

# Display Best Parameters
print("\nBest Hyperparameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print(round(grid_search.best_score_, 4))

# Tuned Model Prediction
best_model = grid_search.best_estimator_

tuned_predictions = best_model.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_true, predictions):

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions))

# Evaluate Default Model
evaluate_model(
    "Default KNN Results",
    y_test,
    default_predictions
)

# Evaluate Tuned Pipeline
evaluate_model(
    "Tuned KNN Pipeline Results",
    y_test,
    tuned_predictions
)

# Compare Both Models
default_accuracy = accuracy_score(y_test, default_predictions)
tuned_accuracy = accuracy_score(y_test, tuned_predictions)

print("\n" + "=" * 45)
print("MODEL COMPARISON")
print("=" * 45)

print("Default KNN Accuracy:", round(default_accuracy, 4))
print("Tuned KNN Accuracy  :", round(tuned_accuracy, 4))

if tuned_accuracy > default_accuracy:
    print("\nTuned Pipeline performed better.")
elif tuned_accuracy < default_accuracy:
    print("\nDefault KNN performed better.")
else:
    print("\nBoth models achieved the same accuracy.")

# Final Explanation
print("\nFeature Scaling Explanation:")
print("StandardScaler puts numerical features on a similar scale.")
print("This prevents large-valued features from dominating KNN.")
print("GridSearchCV selects the best hyperparameters using 5-fold CV.")
print("The tuned model combines scaling and hyperparameter optimization.")

