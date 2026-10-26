
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Create student dataset
data = {
    "Study_Hours": np.random.uniform(1, 10, 1000),
    "Attendance_Percentage": np.random.uniform(40, 100, 1000),
    "Assignment_Score": np.random.randint(20, 101, 1000),
    "Previous_Exam_Score": np.random.randint(20, 101, 1000)
}

df = pd.DataFrame(data)

# Create Pass/Fail target
final_score = (
    df["Study_Hours"] * 4 +
    df["Attendance_Percentage"] * 0.2 +
    df["Assignment_Score"] * 0.2 +
    df["Previous_Exam_Score"] * 0.2
)

df["Result"] = np.where(final_score >= 60, "Pass", "Fail")

# Save dataset
df.to_csv("student_model_comparison.csv", index=False)

# Load dataset
df = pd.read_csv("student_model_comparison.csv")

print("First 10 Records:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Clean dataset
df = df.drop_duplicates()
df = df.dropna()

# Save cleaned dataset
df.to_csv("student_cleaned.csv", index=False)

# Check Pass/Fail distribution
print("\nPass/Fail Distribution:")
print(df["Result"].value_counts())

print("\nPass/Fail Percentage:")
print(df["Result"].value_counts(normalize=True) * 100)

# Separate features and target
X = df.drop("Result", axis=1)
y = df["Result"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Save training and testing datasets
train_data = X_train.copy()
train_data["Result"] = y_train

test_data = X_test.copy()
test_data["Result"] = y_test

train_data.to_csv("student_train.csv", index=False)
test_data.to_csv("student_test.csv", index=False)

print("\nTraining and Testing Datasets Saved Successfully!")

# Pass/Fail Distribution Chart
counts = df["Result"].value_counts().reindex(
    ["Pass", "Fail"], fill_value=0
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    counts.index,
    counts.values,
    color=["green", "red"]
)

plt.title("Student Pass/Fail Distribution")
plt.xlabel("Student Result")
plt.ylabel("Number of Students")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(int(bar.get_height())),
        ha="center"
    )

plt.tight_layout()
plt.savefig("pass_fail_distribution.png")
plt.show()

print("\nWhy use the same train/test split?")
print(
    "Using the same training and testing data ensures "
    "that Logistic Regression, Decision Tree and KNN "
    "are evaluated fairly on identical student records."
)

print("\nDay 53 Task Completed Successfully!")
