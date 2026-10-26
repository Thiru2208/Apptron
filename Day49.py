import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

# Create 50 customer records
data = {
    "Customer_ID": range(1, 51),
    "Age": np.random.randint(18, 65, 50),
    "Monthly_Spending": np.random.randint(1000, 20000, 50),
    "Website_Visits": np.random.randint(1, 30, 50),
    "Purchased": np.random.choice(["Yes", "No"], 50)
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("customer_knn.csv", index=False)

# Load dataset
df = pd.read_csv("customer_knn.csv")

print("First 10 Customer Records:")
print(df.head(10))

# New customer
new_customer = {
    "Age": 25,
    "Monthly_Spending": 5000,
    "Website_Visits": 12
}

print("\nNew Customer:")
print(new_customer)

# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(
        sum((a - b) ** 2 for a, b in zip(point1, point2))
    )

# Calculate distance for each customer
df["Distance"] = df.apply(
    lambda row: euclidean_distance(
        [row["Age"], row["Monthly_Spending"]],
        [new_customer["Age"], new_customer["Monthly_Spending"]]
    ),
    axis=1
)

# Sort customers by distance
sorted_df = df.sort_values(by="Distance")

print("\nCustomers Sorted by Distance:")
print(sorted_df.to_string(index=False))

# Find 3 nearest neighbors
k = 3
nearest_neighbors = sorted_df.head(k)

print("\nThree Nearest Customers:")
print(nearest_neighbors.to_string(index=False))

# Manual KNN prediction
manual_prediction = nearest_neighbors["Purchased"].mode()[0]

print("\nManual KNN Prediction:")
print("Purchased:", manual_prediction)

# Scikit-learn KNN
X = df[["Age", "Monthly_Spending"]]
y = df["Purchased"]

model = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean"
)

model.fit(X, y)

new_data = pd.DataFrame([{
    "Age": new_customer["Age"],
    "Monthly_Spending": new_customer["Monthly_Spending"]
}])

sklearn_prediction = model.predict(new_data)[0]

print("\nScikit-learn KNN Prediction:")
print("Purchased:", sklearn_prediction)

# Compare predictions
print("\nPrediction Comparison:")
print("Manual KNN:", manual_prediction)
print("Scikit-learn KNN:", sklearn_prediction)

if manual_prediction == sklearn_prediction:
    print("Both predictions match!")
else:
    print("Predictions do not match!")

# Bonus: Euclidean distance using 3 features
print("\nBonus: Distance Using 3 Features")

point1 = [
    new_customer["Age"],
    new_customer["Monthly_Spending"],
    new_customer["Website_Visits"]
]

point2 = [
    df.iloc[0]["Age"],
    df.iloc[0]["Monthly_Spending"],
    df.iloc[0]["Website_Visits"]
]

bonus_distance = euclidean_distance(point1, point2)

print("Euclidean Distance:", bonus_distance)