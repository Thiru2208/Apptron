
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# Create dataset with 50 students
data = {
    "Student_ID": range(1, 51),
    "Study_Hours": np.random.randint(1, 11, 50),
    "Attendance_Percentage": np.random.randint(40, 101, 50),
    "Assignment_Score": np.random.randint(30, 101, 50)
}

df = pd.DataFrame(data)

# Create Pass/Fail results
df["Result"] = np.where(
    (df["Study_Hours"] >= 5) &
    (df["Attendance_Percentage"] >= 65) &
    (df["Assignment_Score"] >= 50),
    "Pass",
    "Fail"
)

# Save dataset
df.to_csv("student_knn_data.csv", index=False)

# Load dataset
df = pd.read_csv("student_knn_data.csv")

print("First 5 Records:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Encode target labels
df["Result_Encoded"] = df["Result"].map({
    "Fail": 0,
    "Pass": 1
})

# Select features and target
X = df[[
    "Study_Hours",
    "Attendance_Percentage",
    "Assignment_Score"
]]

y = df["Result_Encoded"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier with K = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict test results
y_pred = knn.predict(X_test)

print("\nActual vs Predicted Results:")

results = pd.DataFrame({
    "Actual": y_test.map({0: "Fail", 1: "Pass"}).values,
    "Predicted": pd.Series(y_pred).map({
        0: "Fail",
        1: "Pass"
    }).values
})

print(results)

# Evaluate model
print("\nModel Accuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Fail", "Pass"],
    zero_division=0
))

# Predict 5 new students
new_students = pd.DataFrame({
    "Study_Hours": [8, 3, 6, 2, 9],
    "Attendance_Percentage": [90, 50, 75, 45, 95],
    "Assignment_Score": [85, 40, 70, 35, 90]
})

new_scaled = scaler.transform(new_students)

new_predictions = knn.predict(new_scaled)

new_students["Predicted_Result"] = [
    "Pass" if result == 1 else "Fail"
    for result in new_predictions
]

print("\nPredictions for 5 New Students:")
print(new_students)

# Save prediction results
new_students.to_csv(
    "student_knn_predictions.csv",
    index=False
)

# Compare different K values
print("\nComparison of K Values:")

for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"K = {k}, Accuracy = {accuracy * 100:.2f}%")

print("\nThree Observations:")
print("1. KNN predicts results using nearby student records.")
print("2. Study hours, attendance, and assignment scores influence predictions.")
print("3. Different K values may produce different accuracy scores.")
