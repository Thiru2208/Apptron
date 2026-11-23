
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Breast Cancer Dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Dataset Shape:", X.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Create Default Random Forest Model
default_model = RandomForestClassifier(random_state=42)

# Train Default Model
start_time = time.time()

default_model.fit(X_train, y_train)

default_time = time.time() - start_time

# Predict using Default Model
default_predictions = default_model.predict(X_test)

# Calculate Default Accuracy
default_accuracy = accuracy_score(y_test, default_predictions)

# Define Hyperparameter Search Space
param_distributions = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Create Random Forest Model
model = RandomForestClassifier(random_state=42)

# Apply Randomized Search Cross Validation
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

# Train Random Search Model
start_time = time.time()

random_search.fit(X_train, y_train)

tuned_time = time.time() - start_time

# Display Best Parameters
print("\nBest Hyperparameters:")
print(random_search.best_params_)

# Display Best Cross Validation Score
print("\nBest Cross Validation Score:")
print(random_search.best_score_)

# Get Best Model
best_model = random_search.best_estimator_

# Predict using Tuned Model
tuned_predictions = best_model.predict(X_test)

# Calculate Tuned Accuracy
tuned_accuracy = accuracy_score(y_test, tuned_predictions)

# Display Model Comparison
print("\n========== MODEL COMPARISON ==========")

print("\nDefault Random Forest Accuracy:",
      round(default_accuracy * 100, 2), "%")

print("Tuned Random Forest Accuracy:",
      round(tuned_accuracy * 100, 2), "%")

# Display Execution Time
print("\n========== EXECUTION TIME ==========")

print("Default Model Time:",
      round(default_time, 4), "seconds")

print("Random Search Time:",
      round(tuned_time, 4), "seconds")

print("\nDay 70 Task Completed Successfully!")
