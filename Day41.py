import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(45)

# Create 1000 customer records
data = {
    "Age": np.random.randint(18, 70, 1000),
    "Income": np.random.randint(25000, 200000, 1000),
    "Website_Visits": np.random.randint(1, 30, 1000),
    "Time_Spent": np.random.randint(1, 120, 1000),
    "Previous_Purchases": np.random.randint(0, 20, 1000),
    "Product_Views": np.random.randint(1, 50, 1000)
}

df = pd.DataFrame(data)

# Create target column
df["Purchased"] = (
    (df["Website_Visits"] > 15) |
    (df["Time_Spent"] > 60) |
    (df["Previous_Purchases"] > 10)
).astype(int)

# Display first 10 records
print("First 10 Records:")
print(df.head(10))

# Features and target
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=2000)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Extract TN, FP, FN, TP
tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Correct and incorrect predictions
correct_predictions = (y_test == y_pred).sum()
incorrect_predictions = (y_test != y_pred).sum()

print("\nCorrect Predictions:", correct_predictions)
print("Incorrect Predictions:", incorrect_predictions)

# Actual vs Predicted table
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(20))

# Confusion Matrix Visualization
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Purchased", "Purchased"]
)

display.plot()
plt.title("Customer Purchase Prediction - Confusion Matrix")
plt.show()