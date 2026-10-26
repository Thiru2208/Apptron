import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)

# Create 1000 loan applications
data = {
    "Age": np.random.randint(21, 65, 1000),
    "Income": np.random.randint(20000, 200000, 1000),
    "Credit_Score": np.random.randint(300, 851, 1000),
    "Loan_Amount": np.random.randint(5000, 150000, 1000),
    "Employment_Years": np.random.randint(0, 31, 1000),
    "Existing_Loans": np.random.randint(0, 6, 1000)
}

df = pd.DataFrame(data)

# Create target column
# Approved = 1, Rejected = 0
df["Loan_Approved"] = (
    (df["Credit_Score"] >= 650) &
    (df["Income"] >= 40000) &
    (df["Loan_Amount"] <= df["Income"] * 2) &
    (df["Employment_Years"] >= 2) &
    (df["Existing_Loans"] <= 3)
).astype(int)

# Save dataset
df.to_csv("loan_approval.csv", index=False)

# Load dataset
df = pd.read_csv("loan_approval.csv")

print("First 10 Records:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Clean dataset
df = df.drop_duplicates()
df = df.dropna()

print("\nLoan Approval Distribution:")
print(df["Loan_Approved"].value_counts())

# Separate features and target
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Train Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# Identify root feature
root_index = model.tree_.feature[0]
root_feature = X.columns[root_index]

print("\nRoot Feature:")
print(root_feature)

print("\nFeatures Appearing Early in the Tree:")
print(importance.head())

# Display decision rules
rules = export_text(
    model,
    feature_names=list(X.columns)
)

print("\nDecision Tree Rules:")
print(rules)

# Final leaf predictions
leaf_nodes = np.where(
    model.tree_.children_left == -1
)[0]

print("\nFinal Leaf Predictions:")

for leaf in leaf_nodes:
    prediction = np.argmax(model.tree_.value[leaf][0])

    print(
        f"Leaf {leaf}:",
        "Approved" if prediction == 1 else "Rejected"
    )

# Visualize Decision Tree
plt.figure(figsize=(22, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Loan Approval Decision Tree")
plt.tight_layout()
plt.show()

# Visualize Feature Importance
plt.figure(figsize=(10, 6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Test new loan application
new_loan = pd.DataFrame({
    "Age": [30],
    "Income": [75000],
    "Credit_Score": [720],
    "Loan_Amount": [50000],
    "Employment_Years": [5],
    "Existing_Loans": [1]
})

prediction = model.predict(new_loan)

print("\nNew Loan Application Prediction:")

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")