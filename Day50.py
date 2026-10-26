
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# Create customer dataset
n = 500

data = {
    "Age": np.random.randint(18, 70, n),
    "Monthly_Spending": np.random.randint(1000, 50000, n),
    "Website_Visits": np.random.randint(1, 30, n),
    "Time_on_Website": np.random.randint(1, 120, n),
    "Previous_Purchases": np.random.randint(0, 20, n)
}

df = pd.DataFrame(data)

# Create purchase target
purchase_score = (
    (df["Monthly_Spending"] > 20000).astype(int) +
    (df["Website_Visits"] > 15).astype(int) +
    (df["Time_on_Website"] > 60).astype(int) +
    (df["Previous_Purchases"] > 8).astype(int)
)

df["Purchased"] = np.where(purchase_score >= 2, "Yes", "No")

df.to_csv("customer_purchase_day50.csv", index=False)

# Load dataset
df = pd.read_csv("customer_purchase_day50.csv")

print("FIRST 10 RECORDS")
print(df.head(10))

print("\nDATASET INFORMATION")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

# Clean dataset
df = df.drop_duplicates()
df = df.dropna()

# Separate features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Split into training, validation and testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42,
    stratify=y_temp
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train KNN with different K values
k_values = [1, 3, 5, 7, 9, 11]

results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    results.append({
        "K": k,
        "Validation_Accuracy": val_accuracy,
        "Test_Accuracy": test_accuracy
    })

results_df = pd.DataFrame(results)

print("\nKNN ACCURACY COMPARISON")
print(results_df.to_string(index=False))

# Select best K using validation accuracy
best_row = results_df.loc[
    results_df["Validation_Accuracy"].idxmax()
]

best_k = int(best_row["K"])

print("\nBEST K VALUE:", best_k)
print("BEST VALIDATION ACCURACY:",
      round(best_row["Validation_Accuracy"] * 100, 2), "%")

# Plot accuracy comparison
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["Validation_Accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.plot(
    results_df["K"],
    results_df["Test_Accuracy"],
    marker="s",
    label="Test Accuracy"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN - K Value vs Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Train final model using best K
X_final_train = np.vstack((X_train, X_val))
y_final_train = pd.concat(
    [y_train, y_val], ignore_index=True
)

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_final_train, y_final_train)

final_predictions = final_model.predict(X_test)

print("\nFINAL MODEL RESULTS")
print("Best K:", best_k)
print("Test Accuracy:",
      round(accuracy_score(y_test, final_predictions) * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, final_predictions))

# Predict purchases for 10 new customers
new_customers = pd.DataFrame({
    "Age": [22, 35, 45, 28, 50, 31, 40, 26, 55, 38],
    "Monthly_Spending": [
        5000, 25000, 40000, 12000, 45000,
        18000, 30000, 8000, 35000, 22000
    ],
    "Website_Visits": [
        5, 18, 25, 8, 28, 12, 20, 6, 22, 16
    ],
    "Time_on_Website": [
        15, 75, 100, 30, 110, 45, 85, 20, 90, 65
    ],
    "Previous_Purchases": [
        1, 10, 15, 3, 18, 5, 12, 2, 14, 9
    ]
})

new_scaled = scaler.transform(new_customers)

new_predictions = final_model.predict(new_scaled)

new_customers["Predicted_Purchase"] = new_predictions

print("\nPREDICTIONS FOR 10 NEW CUSTOMERS")
print(new_customers.to_string(index=False))

# Compare uniform and distance-weighted voting
voting_results = []

for voting in ["uniform", "distance"]:
    voting_model = KNeighborsClassifier(
        n_neighbors=best_k,
        weights=voting
    )

    voting_model.fit(X_final_train, y_final_train)

    predictions = voting_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    voting_results.append({
        "Voting_Method": voting,
        "Accuracy": accuracy
    })

voting_df = pd.DataFrame(voting_results)

print("\nVOTING METHOD COMPARISON")
print(voting_df.to_string(index=False))

# Explain K comparison
other_row = results_df[
    results_df["K"] != best_k
].iloc[0]

print("\nCONCLUSION")
print("Best K selected:", best_k)
print(
    "Validation accuracy:",
    round(best_row["Validation_Accuracy"] * 100, 2), "%"
)
print(
    "Compared with K =", int(other_row["K"]),
    "which achieved",
    round(other_row["Validation_Accuracy"] * 100, 2), "%"
)
print(
    "The selected K performed better on validation data, "
    "indicating a better balance between local sensitivity "
    "and generalization for this dataset."
)

# Save results
results_df.to_csv("day50_accuracy_comparison.csv", index=False)
new_customers.to_csv("day50_predictions.csv", index=False)
voting_df.to_csv("day50_voting_comparison.csv", index=False)

print("\nResults saved successfully!")
