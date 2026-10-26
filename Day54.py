
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Create student dataset if Day 1 dataset is unavailable
np.random.seed(42)

data = {
    "Study_Hours": np.random.uniform(1, 10, 1000),
    "Attendance": np.random.uniform(40, 100, 1000),
    "Assignment_Score": np.random.uniform(30, 100, 1000),
    "Previous_Score": np.random.uniform(20, 100, 1000)
}

df = pd.DataFrame(data)

# Create target: 1 = Pass, 0 = Fail
student_score = (
    df["Study_Hours"] * 4 +
    df["Attendance"] * 0.25 +
    df["Assignment_Score"] * 0.25 +
    df["Previous_Score"] * 0.25
)

df["Pass"] = (student_score >= 70).astype(int)

df.to_csv("student_data.csv", index=False)

print("First 10 Records:")
print(df.head(10))

print("\nDataset Information:")
print(df.info())

print("\nClass Distribution:")
print(df["Pass"].value_counts())

# Features and target
X = df.drop("Pass", axis=1)
y = df["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling for Logistic Regression and KNN
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
confusion_matrices = {}

for name, model in models.items():

    if name == "Decision Tree":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    # Positive class 0 = Fail
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusion_matrices[name] = cm

    print(f"\n{name} Results:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

# Metrics comparison table
results_df = pd.DataFrame(results)

print("\nModel Comparison Table:")
print(results_df.to_string(index=False))

# Save comparison table
results_df.to_csv("day54_metrics.csv", index=False)

# Display confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, cm) in zip(
    axes, confusion_matrices.items()
):
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"]
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig("day54_confusion_matrices.png")
plt.show()

# Best model based on recall for Fail class
best_model = results_df.sort_values(
    by=["Recall", "F1-Score", "Accuracy"],
    ascending=False
).iloc[0]

print("\nBest Model for Identifying Students Who May Fail:")
print(best_model["Model"])

print("\nBest Model Metrics:")
print(best_model.to_string())

# Explain false positives and false negatives
print("\nError Analysis:")
print(
    "False Positive (Fail): Model predicts Fail, "
    "but student actually Passes."
)
print(
    "False Negative (Fail): Model predicts Pass, "
    "but student actually Fails."
)

print("\nConclusion:")
print(
    f"{best_model['Model']} is the best model for "
    "identifying students who may fail based on "
    "its recall, with F1-score and accuracy used "
    "as tie-breakers."
)
print(
    "Recall is important because missing students "
    "who actually fail may prevent them from "
    "receiving additional academic support."
)
