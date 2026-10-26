
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

np.random.seed(42)

output_dir = Path(__file__).resolve().parent

# Generate synthetic customer dataset
n = 1200

data = {
    "Age": np.random.randint(18, 70, n),
    "Monthly_Income": np.random.randint(25000, 200000, n),
    "Website_Visits": np.random.randint(1, 31, n),
    "Time_Spent": np.random.randint(1, 121, n),
    "Previous_Purchases": np.random.randint(0, 21, n),
    "Product_Views": np.random.randint(1, 51, n)
}

df = pd.DataFrame(data)

# Generate purchase outcome using customer behaviour
score = (
    0.04 * df["Website_Visits"]
    + 0.025 * df["Time_Spent"]
    + 0.15 * df["Previous_Purchases"]
    + 0.035 * df["Product_Views"]
    + 0.000008 * df["Monthly_Income"]
    + np.random.normal(0, 1.5, n)
)

df["Purchased"] = (score > np.median(score)).astype(int)

# Save dataset
dataset_path = output_dir / "customer_purchase_day52.csv"
df.to_csv(dataset_path, index=False)

# Load dataset
df = pd.read_csv(dataset_path)

print("\nFIRST 10 RECORDS")
print(df.head(10))

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Clean dataset
df = df.drop_duplicates()
df = df.dropna()

print("\nCLASS DISTRIBUTION")
print(df["Purchased"].value_counts())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Class distribution chart
df["Purchased"].value_counts().sort_index().plot(
    kind="bar",
    color=["tomato", "seagreen"]
)

plt.title("Customer Purchase Distribution")
plt.xlabel("Purchased (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / "day52_class_distribution.png")
plt.show()

# Customer behaviour chart
df.groupby("Purchased")["Website_Visits"].mean().plot(
    kind="bar",
    color=["orange", "steelblue"]
)

plt.title("Average Website Visits by Purchase Outcome")
plt.xlabel("Purchased")
plt.ylabel("Average Website Visits")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / "day52_customer_behavior.png")
plt.show()

# Select features and target
features = [
    "Age",
    "Monthly_Income",
    "Website_Visits",
    "Time_Spent",
    "Previous_Purchases",
    "Product_Views"
]

X = df[features]
y = df["Purchased"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING SAMPLES:", len(X_train))
print("TESTING SAMPLES:", len(X_test))

# Cross-validation for different K values
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for k in k_values:

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    results.append({
        "K": k,
        "Mean_CV_Accuracy": scores.mean(),
        "Std_CV_Accuracy": scores.std()
    })

results_df = pd.DataFrame(results)

print("\nK VALUE COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    output_dir / "day52_k_comparison.csv",
    index=False
)

# Plot K-value comparison
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["K"],
    results_df["Mean_CV_Accuracy"],
    marker="o"
)

plt.title("KNN K-Value Comparison")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Mean Cross-Validation Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "day52_k_comparison.png")
plt.show()

# Select best K using cross-validation
best_k = int(
    results_df.loc[
        results_df["Mean_CV_Accuracy"].idxmax(),
        "K"
    ]
)

print("\nBEST K VALUE:", best_k)

# Build final KNN pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

# Train final model
model.fit(X_train, y_train)

# Test predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nFINAL MODEL EVALUATION")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Purchase", "Purchase"]
)

disp.plot(cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "day52_confusion_matrix.png")
plt.show()

# Error analysis
errors = X_test.copy()
errors["Actual"] = y_test
errors["Predicted"] = y_pred

incorrect = errors[
    errors["Actual"] != errors["Predicted"]
]

print("\nMISCLASSIFIED CUSTOMERS")
print("Total Errors:", len(incorrect))
print(incorrect.head(10))

# Predict 10 new customers
new_customers = pd.DataFrame([
    [22, 35000, 5, 10, 0, 6],
    [30, 85000, 18, 65, 8, 25],
    [45, 120000, 25, 95, 15, 40],
    [19, 28000, 3, 8, 0, 4],
    [35, 95000, 22, 80, 12, 35],
    [55, 150000, 10, 35, 4, 15],
    [27, 65000, 16, 55, 7, 22],
    [40, 110000, 28, 110, 18, 45],
    [24, 45000, 7, 20, 2, 10],
    [50, 135000, 20, 75, 11, 32]
], columns=features)

# Generate new predictions
new_predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)[:, 1]

new_customers["Predicted_Purchase"] = new_predictions
new_customers["Purchase_Probability"] = probabilities.round(3)

new_customers["Result"] = new_customers[
    "Predicted_Purchase"
].map({
    0: "No Purchase",
    1: "Purchase"
})

print("\nNEW CUSTOMER PREDICTIONS")
print(new_customers.to_string(index=False))

# Save predictions
new_customers.to_csv(
    output_dir / "day52_new_customer_predictions.csv",
    index=False
)

# Save short project report
report = f"""
DAY 52 - KNN CUSTOMER PURCHASE PREDICTION

Dataset Type: Synthetic
Total Records: {len(df)}
Training Samples: {len(X_train)}
Testing Samples: {len(X_test)}

Algorithm: K-Nearest Neighbors
Scaling Method: StandardScaler
Cross-Validation: Stratified 5-Fold
Best K: {best_k}

FINAL RESULTS
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}

LIMITATIONS
1. The dataset is synthetic, not real customer data.
2. Real customer behaviour may differ from generated patterns.
3. KNN is sensitive to feature scaling.
4. Prediction performance may change with new data.
5. KNN prediction can be slow for very large datasets.

CONCLUSION
The KNN model predicts customer purchase outcomes using
customer demographic and website interaction features.
The best K was selected using cross-validation on training
data, and final performance was measured on unseen test data.
"""

with open(
    output_dir / "day52_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\nPROJECT COMPLETED SUCCESSFULLY")
print("Dataset, charts, predictions and report saved.")
