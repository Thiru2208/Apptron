import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("customer_purchase.csv")

print("First 10 Records:")
print(df.head(10))

# Separate features and target
X = df[
    [
        "Age",
        "Income",
        "Website_Visits",
        "Time_Spent",
        "Previous_Purchases",
        "Product_Views"
    ]
]

y = df["Purchased"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Create prediction table
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Probability": probabilities
})

print("\nPrediction Results:")
print(results.head(20))


# New Customer Predictions

new_customers = pd.DataFrame({
    "Age": [
        28, 35, 42, 24, 50,
        31, 46, 27, 39, 55
    ],

    "Income": [
        120000, 80000, 150000, 45000, 180000,
        95000, 160000, 60000, 110000, 190000
    ],

    "Website_Visits": [
        8, 18, 22, 5, 25,
        14, 20, 7, 17, 26
    ],

    "Time_Spent": [
        25, 75, 90, 20, 100,
        55, 85, 30, 70, 110
    ],

    "Previous_Purchases": [
        3, 8, 12, 1, 15,
        6, 13, 2, 9, 18
    ],

    "Product_Views": [
        12, 25, 35, 8, 45,
        20, 38, 10, 30, 48
    ]
})

# Predict new customers
new_predictions = model.predict(new_customers)

new_probabilities = model.predict_proba(new_customers)[:, 1]

new_customers["Purchase_Probability"] = new_probabilities
new_customers["Prediction"] = new_predictions

print("\nNew Customer Predictions:")
print(new_customers)


# Display in percentage format
print("\nDetailed New Customer Results:")

for i in range(len(new_customers)):

    probability = new_probabilities[i] * 100

    if new_predictions[i] == 1:
        status = "Purchased"
    else:
        status = "Not Purchased"

    print(f"\nCustomer {i + 1}")
    print(f"Purchase Probability: {probability:.2f}%")
    print(f"Prediction: {status}")

# Bonus - Different Decision Thresholds

print("\nThreshold Experiment:")


thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:

    threshold_predictions = (
        probabilities >= threshold
    ).astype(int)

    print(f"\nThreshold = {threshold}")

    threshold_results = pd.DataFrame({
        "Actual": y_test.values[:10],
        "Probability": probabilities[:10],
        "Prediction": threshold_predictions[:10]
    })

    print(threshold_results)