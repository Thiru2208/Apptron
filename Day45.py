
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)

# Create 2000 customer records
data = {
    "Age": np.random.randint(18, 70, 2000),
    "Income": np.random.randint(25000, 200000, 2000),
    "Website_Visits": np.random.randint(1, 30, 2000),
    "Time_Spent": np.random.randint(1, 120, 2000),
    "Previous_Purchases": np.random.randint(0, 20, 2000),
    "Product_Views": np.random.randint(1, 50, 2000)
}

df = pd.DataFrame(data)

# Create target column
df["Purchased"] = (
    (df["Website_Visits"] > 15) |
    (df["Time_Spent"] > 60) |
    (df["Previous_Purchases"] > 10)
).astype(int)

# Save dataset
df.to_csv("customer_purchase_day45.csv", index=False)

# Load dataset
df = pd.read_csv("customer_purchase_day45.csv")

print("First 10 Records:")
print(df.head(10))

print("\nDataset Information:")
print(df.info())

# Clean dataset
print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()
df = df.dropna()

# Separate features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compare actual and predicted values
results = X_test.copy()
results["Actual"] = y_test
results["Predicted"] = y_pred

print("\nActual vs Predicted:")
print(results[["Actual", "Predicted"]].head(20))

# Identify incorrect predictions
incorrect = results[results["Actual"] != results["Predicted"]]

print("\nIncorrect Predictions:")
print(incorrect)

# Visualize Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Purchased", "Purchased"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Customer Purchase Decision Tree")
plt.tight_layout()
plt.savefig("Day45_Decision_Tree.png", dpi=300)
plt.show()

# Display feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\nFeature Importance:")
print(importance)

# Identify root feature
root_feature_index = model.tree_.feature[0]
root_feature = X.columns[root_feature_index]

print("\nRoot Feature:", root_feature)

# Test 10 new customers
new_customers = pd.DataFrame({
    "Age": [22, 35, 45, 28, 50, 31, 40, 26, 55, 33],
    "Income": [30000, 60000, 90000, 45000, 120000,
               55000, 80000, 35000, 150000, 70000],
    "Website_Visits": [5, 18, 10, 22, 8, 25, 12, 17, 4, 28],
    "Time_Spent": [20, 75, 40, 90, 30, 100, 55, 65, 15, 110],
    "Previous_Purchases": [1, 5, 12, 3, 2, 15, 8, 4, 1, 16],
    "Product_Views": [5, 20, 15, 30, 10, 40, 18, 25, 8, 45]
})

new_predictions = model.predict(new_customers)

new_customers["Actual"] = (
    (new_customers["Website_Visits"] > 15) |
    (new_customers["Time_Spent"] > 60) |
    (new_customers["Previous_Purchases"] > 10)
).astype(int)

new_customers["Predicted"] = new_predictions

print("\n10 New Customer Predictions:")
print(new_customers[["Actual", "Predicted"]].to_string(index=True))

# Bonus: Customer purchase prediction function
def predict_customer_purchase(age, income, website_visits,
                              time_spent, previous_purchases,
                              product_views):

    customer = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Website_Visits": website_visits,
        "Time_Spent": time_spent,
        "Previous_Purchases": previous_purchases,
        "Product_Views": product_views
    }])

    prediction = model.predict(customer)[0]

    if prediction == 1:
        return "Purchased"
    else:
        return "Not Purchased"


print("\nBonus Prediction:")

result = predict_customer_purchase(
    age=25,
    income=50000,
    website_visits=20,
    time_spent=80,
    previous_purchases=5,
    product_views=25
)

print("Customer Purchase Status:", result)
