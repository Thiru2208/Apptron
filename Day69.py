
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Breast Cancer dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Loaded Successfully")
print("Dataset Shape:", X.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Create default Decision Tree model
default_model = DecisionTreeClassifier(random_state=42)

# Train default model
default_model.fit(X_train, y_train)

# Predict using default model
default_predictions = default_model.predict(X_test)

# Calculate default accuracy
default_accuracy = accuracy_score(y_test, default_predictions)

print("\nDefault Model Accuracy:", round(default_accuracy * 100, 2), "%")

# Define hyperparameter grid
param_grid = {
    "max_depth": [2, 3, 5, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

# Create Grid Search
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train Grid Search using training data
print("\nRunning Grid Search...")
grid_search.fit(X_train, y_train)

# Display best parameters
print("\nBest Hyperparameters:")
print(grid_search.best_params_)

# Display best cross-validation score
print("\nBest Cross-Validation Score:",
      round(grid_search.best_score_ * 100, 2), "%")

# Get the best model
best_model = grid_search.best_estimator_

# Predict using tuned model
tuned_predictions = best_model.predict(X_test)

# Calculate tuned accuracy
tuned_accuracy = accuracy_score(y_test, tuned_predictions)

print("\nTuned Model Accuracy:",
      round(tuned_accuracy * 100, 2), "%")

# Compare default and tuned models
print("\nModel Comparison")
print("------------------------------")
print("Default Model Accuracy:",
      round(default_accuracy * 100, 2), "%")

print("Tuned Model Accuracy:",
      round(tuned_accuracy * 100, 2), "%")

# Calculate improvement
improvement = (tuned_accuracy - default_accuracy) * 100

print("\nAccuracy Improvement:",
      round(improvement, 2), "%")

# Conclusion
print("\nConclusion:")

if tuned_accuracy > default_accuracy:
    print("The tuned Decision Tree model performs better than the default model.")
elif tuned_accuracy < default_accuracy:
    print("The default Decision Tree model performs better than the tuned model.")
else:
    print("Both models have the same test accuracy.")
