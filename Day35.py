import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("HOUSE PRICE PREDICTION - MODEL EVALUATION")
print("=" * 60)

# Load dataset
df = pd.read_csv("house_data.csv")

print("\nDataset loaded successfully.")

print("\nFIRST 10 RECORDS")
print("-" * 60)
print(df.head(10))

print("\nDATASET INFORMATION")
print("-" * 60)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Select features and target
features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age"
]

X = df[features]
y = df["Price"]

print("\nFEATURES USED")
print("-" * 60)
print(features)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN-TEST SPLIT")
print("-" * 60)
print("Training records:", len(X_train))
print("Testing records :", len(X_test))

# Create and train model
model = LinearRegression()

model.fit(X_train, y_train)

print("\nMODEL TRAINING")
print("-" * 60)
print("Linear Regression model trained successfully.")

# Make predictions
y_pred = model.predict(X_test)

# Display predictions
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results["Error"] = (
    results["Actual Price"]
    - results["Predicted Price"]
)

print("\nACTUAL VS PREDICTED VALUES")
print("-" * 60)
print(results.round(2))

# Calculate evaluation metrics
mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nMODEL EVALUATION METRICS")
print("=" * 60)

print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# Metric comparison table
metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R²"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]
})

print("\nMETRIC COMPARISON TABLE")
print("-" * 60)
print(metrics.to_string(index=False))

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    s=80
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title(
    "Actual vs Predicted House Prices"
)

plt.grid(True)

plt.tight_layout()

plt.show()

# Residual calculation
residuals = (
    y_test.values
    - y_pred
)

# Residual Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    s=80
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residual")

plt.title(
    "Residual Plot"
)

plt.grid(True)

plt.tight_layout()

plt.show()

# Model Analysis
print("\nMODEL ANALYSIS")
print("=" * 60)

# Typical error
print(
    f"Typical prediction error (MAE): "
    f"{mae:,.2f}"
)

print(
    f"RMSE: {rmse:,.2f}"
)

# R² interpretation
print("\nR² INTERPRETATION")
print("-" * 60)

if r2 >= 0.90:
    print(
        "The model explains more than 90% "
        "of the variation in house prices."
    )

elif r2 >= 0.70:
    print(
        "The model explains a good amount "
        "of the variation in house prices."
    )

elif r2 >= 0:
    print(
        "The model explains only a limited "
        "amount of the variation."
    )

else:
    print(
        "The model performs worse than "
        "the simple baseline."
    )

# Check large errors
absolute_errors = np.abs(residuals)

largest_error = np.max(
    absolute_errors
)

print("\nERROR ANALYSIS")
print("-" * 60)

print(
    f"Largest prediction error: "
    f"{largest_error:,.2f}"
)

# Find possible outliers
mean_error = np.mean(
    absolute_errors
)

possible_outliers = (
    absolute_errors
    > 2 * mean_error
)

outlier_count = np.sum(
    possible_outliers
)

print(
    "Possible large-error observations:",
    outlier_count
)

# Suggestions
print("\nPOSSIBLE MODEL IMPROVEMENTS")
print("-" * 60)

print("1. Add more house records.")
print("2. Add more useful features.")
print("3. Check for outliers.")
print("4. Check for non-linear relationships.")
print("5. Avoid data leakage.")

# BONUS TASK
print("\n" + "=" * 60)
print("BONUS - COMPARING TWO FEATURE SETS")
print("=" * 60)

# Model 1 - Area only
X1 = df[["Area"]]

# Model 2 - All features
X2 = df[
    [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Age"
    ]
]

# Use same train/test indices
train_indices = X_train.index
test_indices = X_test.index

X1_train = X1.loc[train_indices]
X1_test = X1.loc[test_indices]

X2_train = X2.loc[train_indices]
X2_test = X2.loc[test_indices]

y_train_bonus = y.loc[train_indices]
y_test_bonus = y.loc[test_indices]

# Train Model 1
model1 = LinearRegression()

model1.fit(
    X1_train,
    y_train_bonus
)

prediction1 = model1.predict(
    X1_test
)

# Train Model 2
model2 = LinearRegression()

model2.fit(
    X2_train,
    y_train_bonus
)

prediction2 = model2.predict(
    X2_test
)

# Model 1 metrics
mae1 = mean_absolute_error(
    y_test_bonus,
    prediction1
)

mse1 = mean_squared_error(
    y_test_bonus,
    prediction1
)

rmse1 = np.sqrt(mse1)

r2_1 = r2_score(
    y_test_bonus,
    prediction1
)

# Model 2 metrics
mae2 = mean_absolute_error(
    y_test_bonus,
    prediction2
)

mse2 = mean_squared_error(
    y_test_bonus,
    prediction2
)

rmse2 = np.sqrt(mse2)

r2_2 = r2_score(
    y_test_bonus,
    prediction2
)

# Comparison table
comparison = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R²"
    ],

    "Model 1 - Area Only": [
        mae1,
        mse1,
        rmse1,
        r2_1
    ],

    "Model 2 - All Features": [
        mae2,
        mse2,
        rmse2,
        r2_2
    ]
})

print("\nMODEL COMPARISON")
print("-" * 60)

print(
    comparison.to_string(
        index=False
    )
)

print("\nModel 1 Features:")
print(["Area"])

print("\nModel 2 Features:")
print([
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age"
])

# Save results
results.to_csv(
    "day35_predictions.csv",
    index=False
)

metrics.to_csv(
    "day35_metrics.csv",
    index=False
)

comparison.to_csv(
    "day35_model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("day35_predictions.csv")
print("day35_metrics.csv")
print("day35_model_comparison.csv")

print("\nDAY 35 TASK COMPLETED SUCCESSFULLY!")