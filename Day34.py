import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate Student Dataset

np.random.seed(42)

n = 100

data = pd.DataFrame({
    "Study_Hours": np.random.uniform(1, 10, n),
    "Attendance": np.random.uniform(50, 100, n),
    "Previous_Score": np.random.uniform(30, 90, n)
})

noise = np.random.normal(0, 5, n)

data["Final_Score"] = (
    3 * data["Study_Hours"]
    + 0.3 * data["Attendance"]
    + 0.4 * data["Previous_Score"]
    + noise
)

data["Final_Score"] = data["Final_Score"].clip(0, 100)

print("STUDENT SCORE PREDICTION ERROR ANALYZER")

print("\nFirst 10 Student Records:")
print(data.head(10))

data.to_csv("student_scores_day34.csv", index=False)


# Select Features and Target

X = data[["Study_Hours", "Attendance", "Previous_Score"]]
y = data["Final_Score"]


# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Information:")
print("Total Records:", len(data))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Training Completed")


# Calculate Evaluation Metrics

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")


# Create Prediction Error Table

error_table = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
})

error_table["Error"] = (
    error_table["Actual"] - error_table["Predicted"]
)

error_table["Absolute_Error"] = error_table["Error"].abs()

error_table["Squared_Error"] = error_table["Error"] ** 2

print("\nPrediction Error Table:")
print(error_table.round(2).to_string(index=False))

error_table.to_csv(
    "day34_prediction_errors.csv",
    index=False
)


# Error Analysis

average_error = error_table["Absolute_Error"].mean()
largest_error = error_table["Absolute_Error"].max()
smallest_error = error_table["Absolute_Error"].min()

largest_index = error_table["Absolute_Error"].idxmax()
smallest_index = error_table["Absolute_Error"].idxmin()

print("\nError Analysis:")

print(f"Average Prediction Error: {average_error:.4f}")
print(f"Largest Error: {largest_error:.4f}")
print(f"Smallest Error: {smallest_error:.4f}")

print("\nRecord With Largest Error:")
print(error_table.loc[largest_index].round(2))

print("\nRecord With Smallest Error:")
print(error_table.loc[smallest_index].round(2))


# Analyze Effect of Large Errors on MSE

sorted_errors = error_table.sort_values(
    by="Squared_Error",
    ascending=False
)

top_errors = sorted_errors.head(5)

print("\nTop 5 Largest Squared Errors:")

print(
    top_errors[
        ["Actual", "Predicted", "Absolute_Error", "Squared_Error"]
    ].round(2).to_string(index=False)
)

total_squared_error = error_table["Squared_Error"].sum()
top_squared_error = top_errors["Squared_Error"].sum()

contribution = (
    top_squared_error / total_squared_error * 100
    if total_squared_error > 0 else 0
)

print(f"\nTop 5 Errors Contribution: {contribution:.2f}%")

if contribution > 50:
    print("Large errors strongly influence MSE.")
else:
    print("Squared errors are relatively distributed.")


# Actual vs Predicted Graph

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    label="Predictions"
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Student Score")
plt.ylabel("Predicted Student Score")
plt.title("Actual vs Predicted Student Scores")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("day34_actual_vs_predicted.png")
plt.show()


# Absolute Error Graph

plt.figure(figsize=(10, 5))

plt.bar(
    range(1, len(error_table) + 1),
    error_table["Absolute_Error"]
)

plt.xlabel("Test Student")
plt.ylabel("Absolute Prediction Error")
plt.title("Student Prediction Absolute Errors")

plt.grid(axis="y")
plt.tight_layout()

plt.savefig("day34_absolute_errors.png")
plt.show()


# Bonus: Compare Two Linear Regression Models

# Model 1: Study Hours Only

model1 = LinearRegression()

model1.fit(
    X_train[["Study_Hours"]],
    y_train
)

pred1 = model1.predict(
    X_test[["Study_Hours"]]
)


# Model 2: Multiple Features

model2 = LinearRegression()

model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)


# Evaluation Function

def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


m1_mae, m1_mse, m1_rmse = evaluate_model(y_test, pred1)
m2_mae, m2_mse, m2_rmse = evaluate_model(y_test, pred2)


# Model Comparison Table

comparison = pd.DataFrame({
    "Model": [
        "Model 1 - Study Hours",
        "Model 2 - Multiple Features"
    ],
    "MAE": [m1_mae, m2_mae],
    "MSE": [m1_mse, m2_mse],
    "RMSE": [m1_rmse, m2_rmse]
})

print("\nModel Comparison:")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "day34_model_comparison.csv",
    index=False
)


# Identify Model With Lower Errors

for metric in ["MAE", "MSE", "RMSE"]:
    best_index = comparison[metric].idxmin()

    print(
        f"Lower {metric}: "
        f"{comparison.loc[best_index, 'Model']}"
    )


# Model Comparison Graph

metrics = ["MAE", "MSE", "RMSE"]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    [m1_mae, m1_mse, m1_rmse],
    width,
    label="Model 1"
)

plt.bar(
    x + width / 2,
    [m2_mae, m2_mse, m2_rmse],
    width,
    label="Model 2"
)

plt.xticks(x, metrics)

plt.ylabel("Error Value")
plt.title("Linear Regression Model Comparison")

plt.legend()
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("day34_model_comparison.png")
plt.show()

print("\nDay 34 Task Completed Successfully!")