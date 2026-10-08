# ============================================================
# STUDENT SCORE PREDICTOR
# Linear Regression & Prediction Analysis
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CREATE DATASET - 500 STUDENTS
# ============================================================

np.random.seed(42)

n_students = 500

study_hours = np.round(np.random.uniform(1, 10, n_students), 1)
attendance = np.random.randint(50, 101, n_students)
assignment_score = np.random.randint(40, 101, n_students)
previous_score = np.random.randint(35, 101, n_students)

# Create Exam Score
noise = np.random.normal(0, 5, n_students)

exam_score = (
    study_hours * 2.5
    + attendance * 0.20
    + assignment_score * 0.25
    + previous_score * 0.30
    + noise
)

# Keep scores between 0 and 100
exam_score = np.clip(exam_score, 0, 100)

exam_score = np.round(exam_score, 2)


# Create DataFrame
df = pd.DataFrame({
    "Student_ID": range(1, n_students + 1),
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignment_Score": assignment_score,
    "Previous_Score": previous_score,
    "Exam_Score": exam_score
})


print("=" * 70)
print("STUDENT SCORE PREDICTOR")
print("=" * 70)

print("\nFIRST 10 STUDENT RECORDS")
print("-" * 70)
print(df.head(10))


# ============================================================
# 2. SAVE DATASET
# ============================================================

df.to_csv("student_score_dataset.csv", index=False)

print("\nDataset saved as: student_score_dataset.csv")
print("Total student records:", len(df))


# ============================================================
# 3. SELECT INPUT FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Previous_Score"
    ]
]

y = df["Exam_Score"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING AND TESTING DATA")
print("-" * 70)

print("Training records :", len(X_train))
print("Testing records  :", len(X_test))


# ============================================================
# 5. BUILD LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 70)

print(f"Mean Absolute Error (MAE) : {mae:.2f}")
print(f"Mean Squared Error (MSE)  : {mse:.2f}")
print(f"Root Mean Squared Error    : {rmse:.2f}")
print(f"R² Score                   : {r2:.4f}")


# ============================================================
# 8. MODEL COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nMODEL COEFFICIENTS")
print("-" * 70)
print(coefficients)

print("\nIntercept:", round(model.intercept_, 2))


# ============================================================
# 9. CREATE RESULTS DATAFRAME
# ============================================================

results = X_test.copy()

results["Student_ID"] = df.loc[X_test.index, "Student_ID"]
results["Actual_Score"] = y_test
results["Predicted_Score"] = y_pred

# Residual = Actual - Predicted
results["Residual"] = (
    results["Actual_Score"]
    - results["Predicted_Score"]
)

# Absolute error
results["Absolute_Error"] = abs(results["Residual"])

# Reorder columns
results = results[
    [
        "Student_ID",
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Previous_Score",
        "Actual_Score",
        "Predicted_Score",
        "Residual",
        "Absolute_Error"
    ]
]

print("\nSAMPLE PREDICTIONS")
print("-" * 70)
print(results.head(10).round(2))


# ============================================================
# 10. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

# Ideal prediction line
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Exam Scores")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 11. REGRESSION VISUALIZATION
# Study Hours vs Exam Score
# ============================================================

simple_X = df[["Study_Hours"]]
simple_y = df["Exam_Score"]

simple_model = LinearRegression()

simple_model.fit(simple_X, simple_y)

simple_prediction = simple_model.predict(simple_X)

# Sort values so regression line appears correctly
sort_index = np.argsort(df["Study_Hours"])

sorted_hours = df["Study_Hours"].iloc[sort_index]
sorted_predictions = simple_prediction[sort_index]


plt.figure(figsize=(8, 6))

plt.scatter(
    df["Study_Hours"],
    df["Exam_Score"],
    alpha=0.6,
    label="Actual Scores"
)

plt.plot(
    sorted_hours,
    sorted_predictions,
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score - Regression Line")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 12. RESIDUAL / ERROR ANALYSIS
# Residual = Actual - Predicted
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    results["Residual"],
    alpha=0.7
)

# Zero-error reference line
plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Exam Score")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Error Analysis")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 13. RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    results["Residual"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Residual Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 14. TOP 10 LARGEST PREDICTION ERRORS
# ============================================================

largest_errors = results.sort_values(
    by="Absolute_Error",
    ascending=False
).head(10)

print("\nTOP 10 LARGEST PREDICTION ERRORS")
print("-" * 70)

print(
    largest_errors[
        [
            "Student_ID",
            "Actual_Score",
            "Predicted_Score",
            "Residual",
            "Absolute_Error"
        ]
    ].round(2)
)


# ============================================================
# 15. TOP 10 MOST ACCURATE PREDICTIONS
# ============================================================

accurate_predictions = results.sort_values(
    by="Absolute_Error",
    ascending=True
).head(10)

print("\nTOP 10 MOST ACCURATE PREDICTIONS")
print("-" * 70)

print(
    accurate_predictions[
        [
            "Student_ID",
            "Actual_Score",
            "Predicted_Score",
            "Absolute_Error"
        ]
    ].round(2)
)


# ============================================================
# 16. CHECK FOR POSSIBLE OUTLIERS
# ============================================================

error_mean = results["Absolute_Error"].mean()
error_std = results["Absolute_Error"].std()

outlier_limit = error_mean + (2 * error_std)

outliers = results[
    results["Absolute_Error"] > outlier_limit
]

print("\nPOSSIBLE OUTLIERS")
print("-" * 70)

print(f"Outlier error threshold: {outlier_limit:.2f}")

if len(outliers) > 0:

    print(
        outliers[
            [
                "Student_ID",
                "Actual_Score",
                "Predicted_Score",
                "Absolute_Error"
            ]
        ].round(2)
    )

else:
    print("No obvious outliers detected.")


# ============================================================
# 17. PREDICT SCORES FOR FIVE NEW STUDENTS
# ============================================================

new_students = pd.DataFrame({

    "Study_Hours": [
        2,
        4,
        6,
        8,
        9
    ],

    "Attendance": [
        60,
        70,
        80,
        90,
        95
    ],

    "Assignment_Score": [
        50,
        65,
        75,
        85,
        95
    ],

    "Previous_Score": [
        45,
        60,
        70,
        80,
        90
    ]
})


new_predictions = model.predict(new_students)

new_predictions = np.clip(
    new_predictions,
    0,
    100
)

new_students["Predicted_Exam_Score"] = np.round(
    new_predictions,
    2
)

new_students.insert(
    0,
    "New_Student_ID",
    range(1, 6)
)


print("\nPREDICTIONS FOR FIVE NEW STUDENTS")
print("-" * 70)

print(new_students)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results.to_csv(
    "student_prediction_results.csv",
    index=False
)

largest_errors.to_csv(
    "top_10_prediction_errors.csv",
    index=False
)

new_students.to_csv(
    "new_student_predictions.csv",
    index=False
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles Created:")
print("1. student_score_dataset.csv")
print("2. student_prediction_results.csv")
print("3. top_10_prediction_errors.csv")
print("4. new_student_predictions.csv")