
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load cleaned student dataset
file_path = Path(__file__).resolve().parent / "student_data.csv"

if not file_path.exists():
    print("Dataset not found. Creating sample student dataset...")

    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "Study_Hours": np.random.uniform(1, 10, n),
        "Attendance": np.random.uniform(40, 100, n),
        "Assignment_Score": np.random.uniform(30, 100, n),
        "Previous_Score": np.random.uniform(20, 100, n)
    })

    score = (
        df["Study_Hours"] * 4
        + df["Attendance"] * 0.3
        + df["Assignment_Score"] * 0.2
        + df["Previous_Score"] * 0.3
        + np.random.normal(0, 10, n)
    )

    df["Pass"] = (score >= 75).astype(int)
    df.to_csv(file_path, index=False)
else:
    df = pd.read_csv(file_path)

print("\nFirst 10 Records:")
print(df.head(10))

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values and duplicate rows
df = df.dropna().drop_duplicates()

# Features and target
if "Pass" not in df.columns:
    raise ValueError("Dataset must contain a 'Pass' target column.")

X = df.drop(columns=["Pass"])
y = df["Pass"]

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Pass must contain 0 (Fail) and 1 (Pass).")

# Same training and testing data for all models
# Change these settings to match Day 1 if different
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Model pipelines and hyperparameters
models = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1, 10]
        }
    },

    "Decision Tree": {
        "pipeline": Pipeline([
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "params": {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10]
        }
    },

    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"]
        }
    }
}

results = []
trained_models = {}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Train and evaluate each model
for name, config in models.items():

    print("\nModel:", name)

    grid = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["params"],
        cv=cv,
        scoring="recall",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    trained_models[name] = best_model

    y_pred = best_model.predict(X_test)

    # Class 0 represents students who may fail
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
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "CV Recall": round(grid.best_score_, 4),
        "Best Parameters": str(grid.best_params_)
    })

    print("Best Parameters:", grid.best_params_)
    print("Cross-Validation Recall:", round(grid.best_score_, 4))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    ))

# Model comparison table
comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="CV Recall",
    ascending=False
)

print("\nMODEL COMPARISON TABLE:")
print(comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv(
    Path(__file__).resolve().parent / "Day55_Model_Comparison.csv",
    index=False
)

# Recommend model based on validation performance
best_model_name = comparison_df.iloc[0]["Model"]

print("\nRECOMMENDED MODEL:", best_model_name)

print(
    "\nReason: This model achieved the highest cross-validation "
    "recall for students at risk of failing."
)

print(
    "\nAccuracy alone is not enough because failing to identify "
    "an at-risk student can be more costly than incorrectly "
    "flagging a student who will pass."
)

print("\nComparison results saved successfully!")
