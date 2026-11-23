
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Breast Cancer dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Number of Records:", X.shape[0])
print("Number of Features:", X.shape[1])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Different max_depth values
depth_values = [2, 3, 5, 8, 10, None]

results = []

# Train and evaluate Decision Tree models
for depth in depth_values:

    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)

    results.append({
        "Max Depth": depth,
        "Training Accuracy": round(train_accuracy * 100, 2),
        "Testing Accuracy": round(test_accuracy * 100, 2)
    })

# Create comparison DataFrame
results_df = pd.DataFrame(results)

print("\nDecision Tree Model Comparison:")
print(results_df.to_string(index=False))

# Find the best model
best_model = results_df.loc[results_df["Testing Accuracy"].idxmax()]

print("\nBest Model:")
print("Max Depth:", best_model["Max Depth"])
print("Testing Accuracy:", best_model["Testing Accuracy"], "%")

# Explanation
print("\nExplanation:")
print("Training accuracy shows how well the model performs on training data.")
print("Testing accuracy shows how well the model performs on unseen data.")

print("A very deep tree may overfit the training data.")
print("A very shallow tree may underfit the data.")
print("Hyperparameter tuning helps find a suitable max_depth value.")
