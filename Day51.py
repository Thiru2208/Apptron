
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load dataset
file_path = Path(__file__).resolve().parent / "customer_purchase.csv"

if not file_path.exists():
    print("Dataset not found. Creating sample customer dataset...")

    np.random.seed(45)
    n = 1000

    df = pd.DataFrame({
        "Age": np.random.randint(18, 71, n),
        "Income": np.random.randint(25000, 200001, n),
        "Website_Visits": np.random.randint(1, 31, n),
        "Time_Spent": np.random.randint(1, 121, n),
        "Previous_Purchases": np.random.randint(0, 21, n),
        "Product_Views": np.random.randint(1, 51, n)
    })

    df["Purchased"] = (
        (df["Website_Visits"] > 15) |
        (df["Time_Spent"] > 60) |
        (df["Previous_Purchases"] > 10)
    ).astype(int)

    df.to_csv(file_path, index=False)

df = pd.read_csv(file_path)

print("\nFIRST 10 RECORDS")
print(df.head(10))

print("\nDATASET INFORMATION")
print("Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

print("\nCLASS DISTRIBUTION")
print(df["Purchased"].value_counts())
print(df["Purchased"].value_counts(normalize=True) * 100)

# Data preparation
df = df.drop_duplicates()

if df.isnull().any().any():
    df = df.dropna()
    print("\nRows containing missing values removed.")

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

if y.nunique() != 2:
    raise ValueError("Purchased must contain two classes: 0 and 1.")

if y.value_counts().min() < 5:
    raise ValueError("Each class needs at least 5 records.")

# Same split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# Create model pipelines
models = {
    "KNN Without Scaling": Pipeline([
        ("knn", KNeighborsClassifier())
    ]),

    "KNN With StandardScaler": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ]),

    "KNN With MinMaxScaler": Pipeline([
        ("scaler", MinMaxScaler()),
        ("knn", KNeighborsClassifier())
    ])
}

# Cross-validation on training data only
k_values = [3, 5, 7, 9, 11, 15, 21]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
best_models = {}

print("\nCROSS-VALIDATION RESULTS")

for name, pipeline in models.items():

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={"knn__n_neighbors": k_values},
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_

    print("\nModel:", name)
    print("Best K:", grid.best_params_["knn__n_neighbors"])
    print("Best CV F1:", round(grid.best_score_, 4))

    results.append({
        "Model": name,
        "Best K": grid.best_params_["knn__n_neighbors"],
        "CV F1": grid.best_score_
    })

cv_results = pd.DataFrame(results)

print("\nMODEL VALIDATION COMPARISON")
print(cv_results.to_string(index=False))

# Select model using validation performance
best_model_name = cv_results.loc[
    cv_results["CV F1"].idxmax(), "Model"
]

best_model = best_models[best_model_name]

print("\nSELECTED MODEL:", best_model_name)

# Evaluate models on test data
# Test data was not used for K selection or model selection
test_results = []

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, model) in zip(axes, best_models.items()):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    test_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nMODEL:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(name)

plt.tight_layout()
plt.show()

test_df = pd.DataFrame(test_results)

print("\nFINAL TEST COMPARISON")
print(test_df.round(4).to_string(index=False))

# Detailed report for selected model
selected_predictions = best_model.predict(X_test)

print("\nSELECTED MODEL CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    selected_predictions,
    zero_division=0
))

# Compare scaling performance
unscaled_f1 = test_df.loc[
    test_df["Model"] == "KNN Without Scaling", "F1 Score"
].iloc[0]

scaled_f1 = test_df.loc[
    test_df["Model"] == "KNN With StandardScaler", "F1 Score"
].iloc[0]

print("\nSCALING ANALYSIS")

if scaled_f1 > unscaled_f1:
    print("StandardScaler improved KNN test F1-score.")
elif scaled_f1 < unscaled_f1:
    print("KNN without scaling achieved a higher test F1-score.")
else:
    print("Both models achieved the same test F1-score.")

print(
    "Scaling gives features comparable numerical ranges, "
    "so large-valued features do not automatically dominate "
    "KNN distance calculations."
)

print("\nFINAL SELECTED MODEL:", best_model_name)
print("Best K:", best_model.named_steps["knn"].n_neighbors)
