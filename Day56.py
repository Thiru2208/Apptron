
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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

np.random.seed(42)

output_dir = Path("Day56_Results")
output_dir.mkdir(exist_ok=True)

# Create student dataset
n = 1000

df = pd.DataFrame({
    "Study_Hours": np.random.uniform(1, 10, n),
    "Attendance": np.random.uniform(40, 100, n),
    "Assignment_Score": np.random.uniform(30, 100, n),
    "Previous_Score": np.random.uniform(30, 100, n)
})

# Create pass/fail target
score = (
    df["Study_Hours"] * 4
    + df["Attendance"] * 0.25
    + df["Assignment_Score"] * 0.25
    + df["Previous_Score"] * 0.20
    + np.random.normal(0, 8, n)
)

df["Pass"] = (score >= 65).astype(int)

df.to_csv(output_dir / "student_data.csv", index=False)

print("FIRST 10 RECORDS")
print(df.head(10))

print("\nCLASS DISTRIBUTION")
print(df["Pass"].value_counts())

X = df.drop("Pass", axis=1)
y = df["Pass"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
}

results = []
predictions = {}

for name, model in models.items():

    print(f"\nMODEL: {name}")

    model.fit(X_train, y_train)

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - start

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Class 0 = Fail, the important class
    precision = precision_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    gap = train_accuracy - test_accuracy

    results.append({
        "Model": name,
        "Train_Accuracy": train_accuracy,
        "Test_Accuracy": test_accuracy,
        "Precision_Fail": precision,
        "Recall_Fail": recall,
        "F1_Fail": f1,
        "Overfitting_Gap": gap,
        "Prediction_Time": prediction_time,
        "Missed_Failures": int(cm[0, 1]),
        "False_Fail_Alerts": int(cm[1, 0])
    })

    predictions[name] = y_pred

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Fail Precision: {precision:.4f}")
    print(f"Fail Recall: {recall:.4f}")
    print(f"Fail F1-Score: {f1:.4f}")
    print(f"Overfitting Gap: {gap:.4f}")
    print(f"Prediction Time: {prediction_time:.6f} seconds")
    print("Confusion Matrix:")
    print(cm)

results_df = pd.DataFrame(results)

results_df.to_csv(
    output_dir / "model_results.csv",
    index=False
)

print("\nMODEL COMPARISON TABLE")
print(results_df.round(4).to_string(index=False))

# Chart 1: Test accuracy comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["Test_Accuracy"]
)

plt.title("Test Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(output_dir / "accuracy_comparison.png")
plt.show()

# Chart 2: Precision, Recall, F1-score
x = np.arange(len(results_df))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    results_df["Precision_Fail"],
    width,
    label="Precision"
)

plt.bar(
    x,
    results_df["Recall_Fail"],
    width,
    label="Recall"
)

plt.bar(
    x + width,
    results_df["F1_Fail"],
    width,
    label="F1-Score"
)

plt.xticks(x, results_df["Model"])
plt.title("Precision, Recall and F1-Score (Fail Class)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "metrics_comparison.png")
plt.show()

# Chart 3: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, y_pred) in zip(axes, predictions.items()):

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"]
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig(output_dir / "confusion_matrices.png")
plt.show()

# Chart 4: Training vs testing accuracy
x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    results_df["Train_Accuracy"],
    width,
    label="Training"
)

plt.bar(
    x + width / 2,
    results_df["Test_Accuracy"],
    width,
    label="Testing"
)

plt.xticks(x, results_df["Model"])
plt.title("Training vs Testing Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "train_test_comparison.png")
plt.show()

# Chart 5: Prediction time
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["Prediction_Time"] * 1000
)

plt.title("Prediction Time Comparison")
plt.xlabel("Models")
plt.ylabel("Prediction Time (milliseconds)")
plt.tight_layout()
plt.savefig(output_dir / "prediction_time.png")
plt.show()

# Overfitting analysis
print("\nOVERFITTING ANALYSIS")

for _, row in results_df.iterrows():

    gap = row["Overfitting_Gap"]

    if gap > 0.10:
        status = "Possible Overfitting"
    elif row["Train_Accuracy"] < 0.70 and row["Test_Accuracy"] < 0.70:
        status = "Possible Underfitting"
    else:
        status = "No Strong Overfitting Indication"

    print(f"{row['Model']}: {status} (Gap: {gap:.4f})")

# Recommend model for identifying failing students
# Prioritize recall for Fail, followed by F1 and accuracy
ranked = results_df.sort_values(
    by=["Recall_Fail", "F1_Fail", "Test_Accuracy"],
    ascending=False
)

best = ranked.iloc[0]

print("\nRECOMMENDED MODEL")
print("Model:", best["Model"])
print("Fail Recall:", round(best["Recall_Fail"], 4))
print("Fail F1-Score:", round(best["F1_Fail"], 4))
print("Test Accuracy:", round(best["Test_Accuracy"], 4))
print("Missed Failures:", int(best["Missed_Failures"]))

# Five observations
observations = [
    f"1. {results_df.loc[results_df['Test_Accuracy'].idxmax(), 'Model']} achieved the highest test accuracy.",
    f"2. {results_df.loc[results_df['Recall_Fail'].idxmax(), 'Model']} achieved the highest recall for failing students.",
    f"3. {results_df.loc[results_df['F1_Fail'].idxmax(), 'Model']} achieved the highest F1-score for the Fail class.",
    f"4. {results_df.loc[results_df['Overfitting_Gap'].idxmax(), 'Model']} showed the largest training-testing accuracy gap.",
    f"5. {best['Model']} is recommended because it prioritizes detecting students who may fail."
]

print("\nFIVE OBSERVATIONS")

for observation in observations:
    print(observation)

# Save short analysis report
report = [
    "DAY 56 - MODEL RESULTS ANALYSIS",
    "",
    "MODEL RESULTS",
    results_df.round(4).to_string(index=False),
    "",
    "FIVE OBSERVATIONS",
    *observations,
    "",
    "RECOMMENDED MODEL",
    f"Model: {best['Model']}",
    f"Fail Recall: {best['Recall_Fail']:.4f}",
    f"Fail F1-Score: {best['F1_Fail']:.4f}",
    f"Test Accuracy: {best['Test_Accuracy']:.4f}",
    f"Missed Failures: {int(best['Missed_Failures'])}",
    "",
    "Reason: High recall for the Fail class helps identify",
    "students at risk of failing while minimizing missed failures."
]

(output_dir / "analysis_report.txt").write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\nAll charts, CSV files and report saved successfully!")
print("Output Folder:", output_dir.resolve())
