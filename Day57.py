
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


OUTPUT = Path(__file__).resolve().parent / "Day57_results"
OUTPUT.mkdir(exist_ok=True)

DATA_FILE = OUTPUT / "student_support_dataset.csv"

rng = np.random.default_rng(42)
n = 1000

study_hours = rng.uniform(0.5, 10, n)
attendance = rng.uniform(35, 100, n)
assignment = rng.uniform(25, 100, n)
previous = rng.uniform(20, 100, n)

score = (
    0.29 * study_hours / 10
    + 0.27 * attendance / 100
    + 0.23 * assignment / 100
    + 0.21 * previous / 100
    + rng.normal(0, 0.09, n)
)

needs_support = (score < 0.57).astype(int)

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignment_Score": assignment,
    "Previous_Score": previous,
    "Needs_Support": needs_support
})

for col in ["Attendance", "Assignment_Score"]:
    df.loc[
        rng.choice(n, size=12, replace=False), col
    ] = np.nan

df.to_csv(DATA_FILE, index=False)

print("STUDENT SUPPORT PREDICTION SYSTEM")

print("\nFirst 10 Records:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isna().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["Needs_Support"].value_counts())

print("\nStatistical Summary:")
print(df.describe().round(2))


df = df.drop_duplicates().copy()

features = [
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Previous_Score"
]

X = df[features]
y = df["Needs_Support"]

# 1 = Needs additional academic support
# 0 = Does not need additional support


fig, axs = plt.subplots(2, 3, figsize=(14, 8))

for ax, col in zip(axs.flat[:4], features):
    ax.hist(df[col].dropna(), bins=20, edgecolor="black")
    ax.set_title(col)

axs[1, 1].bar(
    ["No Support (0)", "Support (1)"],
    [(y == 0).sum(), (y == 1).sum()]
)

axs[1, 1].set_title("Target Distribution")
axs[1, 2].axis("off")

fig.suptitle("Student Dataset Exploration")
fig.tight_layout()

fig.savefig(OUTPUT / "eda.png", dpi=150)
plt.close(fig)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


preprocess_scaled = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        features
    )
])

preprocess_tree = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        features
    )
])


models = {
    "Logistic Regression": (
        Pipeline([
            ("preprocess", preprocess_scaled),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=42
            ))
        ]),
        {
            "model__C": [0.1, 1, 10]
        }
    ),

    "Decision Tree": (
        Pipeline([
            ("preprocess", preprocess_tree),
            ("model", DecisionTreeClassifier(
                random_state=42
            ))
        ]),
        {
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_leaf": [1, 5, 10]
        }
    ),

    "KNN": (
        Pipeline([
            ("preprocess", preprocess_scaled),
            ("model", KNeighborsClassifier())
        ]),
        {
            "model__n_neighbors": [3, 5, 7, 11, 15],
            "model__weights": ["uniform", "distance"]
        }
    )
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rows = []
matrices = {}
reports = []

for name, (pipeline, params) in models.items():

    print(f"\nTraining {name}...")

    search = GridSearchCV(
        pipeline,
        params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    best = search.best_estimator_

    train_pred = best.predict(X_train)

    start = time.perf_counter()
    test_pred = best.predict(X_test)
    prediction_seconds = time.perf_counter() - start

    train_f1 = f1_score(
        y_train, train_pred, zero_division=0
    )

    test_f1 = f1_score(
        y_test, test_pred, zero_division=0
    )

    rows.append({
        "Model": name,
        "CV_F1": search.best_score_,
        "Train_Accuracy": accuracy_score(y_train, train_pred),
        "Test_Accuracy": accuracy_score(y_test, test_pred),
        "Test_Precision": precision_score(
            y_test, test_pred, zero_division=0
        ),
        "Test_Recall": recall_score(
            y_test, test_pred, zero_division=0
        ),
        "Test_F1": test_f1,
        "Train_F1": train_f1,
        "Train_Test_F1_Gap": train_f1 - test_f1,
        "Tuning_Time_Sec": training_seconds,
        "Predict_Time_Sec": prediction_seconds,
        "Best_Params": str(search.best_params_)
    })

    matrices[name] = confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1]
    )

    reports.append(
        f"{name}\n"
        f"Best parameters: {search.best_params_}\n"
        + classification_report(
            y_test, test_pred, zero_division=0
        )
    )

    print("\nBest Parameters:", search.best_params_)
    print("Cross-Validation F1:", round(search.best_score_, 4))
    print("Test Accuracy:", round(accuracy_score(y_test, test_pred), 4))
    print("Test Recall:", round(recall_score(y_test, test_pred), 4))
    print("Test F1:", round(test_f1, 4))


results = pd.DataFrame(rows).sort_values(
    "CV_F1",
    ascending=False
).reset_index(drop=True)

results.to_csv(
    OUTPUT / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON RESULTS")

print(
    results[
        [
            "Model",
            "CV_F1",
            "Test_Accuracy",
            "Test_Precision",
            "Test_Recall",
            "Test_F1",
            "Train_Test_F1_Gap"
        ]
    ].round(3).to_string(index=False)
)


fig, axs = plt.subplots(1, 3, figsize=(14, 4))

for ax, (name, cm) in zip(axs, matrices.items()):

    ConfusionMatrixDisplay(
        cm,
        display_labels=["No Support", "Support"]
    ).plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(name)

fig.tight_layout()

fig.savefig(
    OUTPUT / "confusion_matrices.png",
    dpi=150
)

plt.close(fig)


metrics = [
    "Test_Accuracy",
    "Test_Precision",
    "Test_Recall",
    "Test_F1"
]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05)
)

ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")

ax.legend(
    ["Accuracy", "Precision", "Recall", "F1"],
    loc="lower right"
)

plt.xticks(rotation=12, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT / "performance_dashboard.png",
    dpi=150
)

plt.close()


ax = results.set_index("Model")[
    ["Train_F1", "Test_F1"]
].plot(
    kind="bar",
    figsize=(9, 5),
    ylim=(0, 1.05)
)

ax.set_title("Training vs Testing F1 Score")
ax.set_ylabel("F1 Score")

plt.xticks(rotation=12, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT / "overfitting_comparison.png",
    dpi=150
)

plt.close()


winner = results.iloc[0]

print("\nFINAL MODEL RECOMMENDATION")

print("Recommended Model:", winner["Model"])
print("Cross-Validation F1:", round(winner["CV_F1"], 4))
print("Test Accuracy:", round(winner["Test_Accuracy"], 4))
print("Test Recall:", round(winner["Test_Recall"], 4))
print("Test F1:", round(winner["Test_F1"], 4))

print(
    "\nReason: Highest cross-validation F1 score "
    "for identifying students who need academic support."
)

print(
    "\nNote: Synthetic data is used. Results must be "
    "validated on real student data before deployment."
)


report = (
    "DAY 57 - STUDENT SUPPORT PREDICTION REPORT\n\n"
    "Dataset: 1000 synthetic student records\n"
    "Target: 1 = Needs academic support\n\n"
    "Model selection: 5-fold stratified cross-validation\n\n"
    + results.to_string(index=False)
    + "\n\n"
    + "\n\n".join(reports)
    + "\n\nRECOMMENDED MODEL: "
    + str(winner["Model"])
    + "\nReason: Highest training cross-validation F1 score."
    + "\n\nLimitations: Synthetic dataset, simplified labels, "
      "and no fairness or real-world validation."
)

(OUTPUT / "final_report.txt").write_text(
    report,
    encoding="utf-8"
)

(OUTPUT / "README.txt").write_text(
    "Day57 - Student Support Prediction System\n"
    "Install: python -m pip install numpy pandas matplotlib scikit-learn\n"
    "Run: python Day57.py\n"
    "All results are saved in the Day57_results folder.\n",
    encoding="utf-8"
)

print("\nAll results saved successfully!")
print("Output Folder:", OUTPUT)
