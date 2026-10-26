
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)

# Part 1 - Load Dataset

filename = "heart.csv"

if os.path.exists(filename):
    df = pd.read_csv(filename)
    print("Existing heart disease dataset loaded.")

else:
    print("Dataset not found. Creating 1200 simulated records.")

    n = 1200

    df = pd.DataFrame({
        "Age": np.random.randint(25, 81, n),
        "Sex": np.random.randint(0, 2, n),
        "ChestPainType": np.random.randint(0, 4, n),
        "RestingBP": np.random.randint(90, 181, n),
        "Cholesterol": np.random.randint(140, 351, n),
        "MaxHeartRate": np.random.randint(80, 201, n),
        "ExerciseAngina": np.random.randint(0, 2, n),
        "FastingBS": np.random.randint(0, 2, n)
    })

    # Artificial labels for educational practice only
    risk = (
        (df["Age"] > 55).astype(int) * 2
        + (df["Cholesterol"] > 240).astype(int)
        + (df["RestingBP"] > 140).astype(int)
        + (df["MaxHeartRate"] < 120).astype(int)
        + df["ExerciseAngina"] * 2
        + (df["ChestPainType"] == 3).astype(int)
        + np.random.normal(0, 1.5, n)
    )

    df["HeartDisease"] = (risk >= 4).astype(int)
    df.to_csv(filename, index=False)
    print("Simulated dataset saved as heart.csv")

print("\nFirst 10 Records:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Data Cleaning

df = df.drop_duplicates().copy()

target = "HeartDisease"

if target not in df.columns:
    raise ValueError("Target column 'HeartDisease' not found.")

df = df.dropna(subset=[target])

# Separate Features and Target

X = df.drop(columns=[target])
y = df[target]

# Encode categorical features
X = pd.get_dummies(X, drop_first=False, dtype=int)

# Fill missing feature values
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

if y.nunique() != 2:
    raise ValueError("Target must contain two classes.")

print("\nDataset After Cleaning:")
print(X.shape)

print("\nTarget Distribution:")
print(y.value_counts())

# Part 2 - Exploratory Data Analysis

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.countplot(x=y)
plt.title("Heart Disease Class Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

if "Age" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.histplot(df["Age"], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.show()

if "Cholesterol" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.histplot(df["Cholesterol"], bins=20, kde=True)
    plt.title("Cholesterol Distribution")
    plt.xlabel("Cholesterol")
    plt.tight_layout()
    plt.show()

if "Age" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=target, y="Age")
    plt.title("Age vs Heart Disease")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 7))
correlation_data = X.copy()
correlation_data[target] = y

sns.heatmap(
    correlation_data.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nFive Data Insights:")

print("1. Target class distribution:")
print(y.value_counts(normalize=True).round(3))

if "Age" in df.columns:
    print("\n2. Average age by target:")
    print(df.groupby(target)["Age"].mean().round(2))

if "Cholesterol" in df.columns:
    print("\n3. Average cholesterol by target:")
    print(df.groupby(target)["Cholesterol"].mean().round(2))

if "RestingBP" in df.columns:
    print("\n4. Average resting BP by target:")
    print(df.groupby(target)["RestingBP"].mean().round(2))

print("\n5. Strongest feature correlations with target:")
print(
    correlation_data.corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
    .head(5)
)

# Part 3 - Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Train Initial Decision Tree

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\nInitial Decision Tree Results")
print("Training Accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy:", round(test_accuracy * 100, 2), "%")

# Part 4 - Model Evaluation

cm = confusion_matrix(y_test, test_pred, labels=model.classes_)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, test_pred, zero_division=0))

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

comparison = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": test_pred
})

print("\nActual vs Predicted:")
print(comparison.head(20))

# Part 5 - Overfitting Experiment

depths = [2, 3, 5, 8, 10]

results = []
models = {}

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)

    results.append({
        "Max Depth": depth,
        "Training Accuracy": train_acc,
        "Testing Accuracy": test_acc,
        "Accuracy Gap": train_acc - test_acc
    })

    models[depth] = tree

results_df = pd.DataFrame(results)

print("\nDecision Tree Depth Comparison:")
print(results_df.round(4).to_string(index=False))

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Max Depth"],
    results_df["Training Accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    results_df["Max Depth"],
    results_df["Testing Accuracy"],
    marker="s",
    label="Testing Accuracy"
)

plt.title("Decision Tree Overfitting Analysis")
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Select best model using testing accuracy
# For this educational practical only

best_row = results_df.sort_values(
    by=["Testing Accuracy", "Accuracy Gap", "Max Depth"],
    ascending=[False, True, True]
).iloc[0]

best_depth = int(best_row["Max Depth"])
best_model = models[best_depth]

print("\nBest Tree Depth:", best_depth)

print("\nUnderfitting Candidate:")
print(
    results_df.sort_values("Training Accuracy")
    .head(1).to_string(index=False)
)

print("\nOverfitting Candidate:")
print(
    results_df.sort_values("Accuracy Gap", ascending=False)
    .head(1).to_string(index=False)
)

print("\nBest Generalizing Candidate:")
print(results_df.loc[[best_row.name]].to_string(index=False))

# Part 6 - Final Tree Visualization

plt.figure(figsize=(22, 12))

plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=[str(c) for c in best_model.classes_],
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=4
)

plt.title(f"Final Decision Tree - Max Depth {best_depth}")
plt.tight_layout()
plt.savefig("Day47_Decision_Tree.png", dpi=150)
plt.show()

print("\nDecision Tree Rules:")

rules = export_text(
    best_model,
    feature_names=list(X.columns),
    max_depth=best_depth
)

print(rules)

print("\nFive Decision Rules:")

def extract_paths(tree_model, feature_names, limit=5):
    tree = tree_model.tree_
    paths = []

    def traverse(node, conditions):
        if tree.feature[node] == -2:
            prediction = tree_model.classes_[
                np.argmax(tree.value[node][0])
            ]
            paths.append((conditions, prediction))
            return

        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]

        traverse(
            tree.children_left[node],
            conditions + [f"{feature} <= {threshold:.2f}"]
        )

        traverse(
            tree.children_right[node],
            conditions + [f"{feature} > {threshold:.2f}"]
        )

    traverse(0, [])

    for i, (conditions, prediction) in enumerate(paths[:limit], 1):
        print(
            f"{i}. IF {' AND '.join(conditions)} "
            f"THEN Heart Disease = {prediction}"
        )

extract_paths(best_model, list(X.columns))

# Part 7 - Final Report

final_train_accuracy = best_model.score(X_train, y_train)
final_test_accuracy = best_model.score(X_test, y_test)

gap = final_train_accuracy - final_test_accuracy

print("\nFINAL PROJECT REPORT")

print("Dataset Size:", len(df))
print("Number of Features:", X.shape[1])
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print(
    "Training Accuracy:",
    round(final_train_accuracy * 100, 2), "%"
)

print(
    "Testing Accuracy:",
    round(final_test_accuracy * 100, 2), "%"
)

print("Best Tree Depth:", best_depth)

print(
    "Overfitting Observed:",
    "Yes" if gap > 0.05 else "No significant gap"
)

print("Final Model: Decision Tree Classifier")

print(
    "Why Selected: Highest testing accuracy among "
    "tested depths, with smaller accuracy gap "
    "used to break ties."
)

print("\nImportant Observations:")
print("1. Decision Trees can classify tabular records.")
print("2. Greater tree depth increases model complexity.")
print("3. Deep trees may overfit the training dataset.")
print("4. Training and testing accuracy should be compared.")
print("5. The final model balances accuracy and complexity.")

print(
    "\nNote: This is an educational machine learning "
    "exercise, not a medical diagnostic system."
)
