import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# Create 500 student records
data = {
    "Study_Hours": np.random.uniform(1, 10, 500),
    "Attendance": np.random.randint(40, 101, 500),
    "Assignment_Score": np.random.randint(30, 101, 500),
    "Previous_Score": np.random.randint(20, 101, 500)
}

df = pd.DataFrame(data)

# Create target column
df["Pass"] = (
    (df["Study_Hours"] > 5) &
    (df["Attendance"] > 75) &
    (df["Assignment_Score"] > 50)
).astype(int)

# Save dataset
df.to_csv("student_decision_tree.csv", index=False)

# Load dataset
df = pd.read_csv("student_decision_tree.csv")

print("First 10 Records:")
print(df.head(10))

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nPass/Fail Distribution:")
print(df["Pass"].value_counts())

# Features and target
X = df.drop("Pass", axis=1)
y = df["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Compare actual and predicted values
results = X_test.copy()
results["Actual"] = y_test
results["Predicted"] = y_pred

print("\nActual vs Predicted:")
print(results.head(10))

# Predict new student
new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Attendance": [85],
    "Assignment_Score": [78],
    "Previous_Score": [70]
})

prediction = model.predict(new_student)

print("\nNew Student Prediction:")

if prediction[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")

# Visualize Decision Tree
plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Student Pass/Fail Decision Tree")
plt.tight_layout()
plt.show()

# Bonus: Trace 5 students
print("\nDecision Paths for 5 Students:")

sample_students = X_test.head(5)
decision_paths = model.decision_path(sample_students)

tree = model.tree_

for i in range(5):
    student = sample_students.iloc[i:i+1]
    prediction = model.predict(student)[0]

    print(f"\nStudent {i+1}")

    node_indices = decision_paths.indices[
        decision_paths.indptr[i]:decision_paths.indptr[i+1]
    ]

    for node_id in node_indices:
        feature_index = tree.feature[node_id]

        if feature_index >= 0:
            feature_name = X.columns[feature_index]
            threshold = tree.threshold[node_id]
            value = student.iloc[0, feature_index]

            if value <= threshold:
                print(f"{feature_name} <= {threshold:.2f}: Yes")
            else:
                print(f"{feature_name} <= {threshold:.2f}: No")

    print("Prediction:", "Pass" if prediction == 1 else "Fail")