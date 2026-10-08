from turtle import color

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

study_hours = np.random.uniform(1, 10, 500)
attendance = np.random.uniform(40, 100, 500)
assignment_score = np.random.uniform(30, 100, 500)
previous_score = np.random.uniform(30, 100, 500)

pass_fail = (
    (study_hours >= 5) &
    (attendance >= 70) &
    (assignment_score >= 50) &
    (previous_score >= 50)
).astype(int)

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignment_Score": assignment_score,
    "Previous_Score": previous_score,
    "Pass": pass_fail
})

print("First 10 Records:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nPass / Fail Count:")
print(df["Pass"].value_counts())

plt.scatter(df["Study_Hours"], df["Pass"], color="red")

plt.xlabel("Study Hours")
plt.ylabel("Pass / Fail")
plt.title("Study Hours vs Pass/Fail")

plt.yticks([0, 1], ["Fail", "Pass"])

plt.show()

plt.scatter(df["Attendance"], df["Pass"], color="green")

plt.xlabel("Attendance")
plt.ylabel("Pass / Fail")
plt.title("Attendance vs Pass/Fail")

plt.yticks([0, 1], ["Fail", "Pass"])

plt.show()

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Previous_Score"
    ]
]

y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nFeatures:")
print("Study Hours")
print("Attendance")
print("Assignment Score")
print("Previous Score")

print("\nTarget:")
print("Pass")

print("\nClasses:")
print("0 = Fail")
print("1 = Pass")

print("\nWhy Classification?")
print("The output is a category: Pass or Fail.")

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Attendance": [85],
    "Assignment_Score": [75],
    "Previous_Score": [70]
})

prediction = model.predict(new_student)

probability = model.predict_proba(new_student)

print("\nNew Student Prediction:")

if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")

print(
    "Pass Probability:",
    round(probability[0][1] * 100, 2),
    "%"
)

print("\nRule Based Prediction:")

if (
    new_student["Study_Hours"].iloc[0] >= 5
    and new_student["Attendance"].iloc[0] >= 70
):
    print("PASS")
else:
    print("FAIL")