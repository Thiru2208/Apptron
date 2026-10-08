import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create sample dataset if CSV does not exist
if not os.path.exists("house_prices.csv"):
    np.random.seed(42)

    n = 1200

    area = np.random.randint(600, 4000, n)
    bedrooms = np.random.randint(1, 7, n)
    bathrooms = np.random.randint(1, 5, n)
    floors = np.random.randint(1, 4, n)
    age = np.random.randint(0, 50, n)
    parking = np.random.randint(0, 4, n)
    distance = np.round(np.random.uniform(1, 30, n), 2)

    price = (
        area * 12000
        + bedrooms * 1500000
        + bathrooms * 1200000
        + floors * 1000000
        - age * 150000
        + parking * 800000
        - distance * 200000
        + np.random.normal(0, 1500000, n)
    )

    df = pd.DataFrame({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Age": age,
        "Parking": parking,
        "Distance": distance,
        "Price": price
    })

    df.to_csv("house_prices.csv", index=False)

    print("Sample dataset created successfully.\n")


# Load dataset
df = pd.read_csv("house_prices.csv")

print("=" * 60)
print("HOUSE PRICE PREDICTION - MINI PROJECT")
print("=" * 60)


# Task 1 - Data Loading
print("\nFIRST 5 RECORDS")
print("-" * 60)
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# Task 2 - Data Cleaning
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)

df = df.drop_duplicates()

numeric_columns = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Age",
    "Parking",
    "Distance",
    "Price"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()

# Remove invalid values
df = df[
    (df["Area"] > 0) &
    (df["Bedrooms"] > 0) &
    (df["Bathrooms"] > 0) &
    (df["Floors"] > 0) &
    (df["Age"] >= 0) &
    (df["Parking"] >= 0) &
    (df["Distance"] >= 0) &
    (df["Price"] > 0)
]

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nCleaning completed successfully.")


# Task 3 - Exploratory Data Analysis
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# 1. Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], kde=True)

plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# 2. Area vs Price
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="Area",
    y="Price",
    data=df
)

plt.title("Area vs House Price")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")

plt.tight_layout()
plt.show()


# 3. Bedrooms vs Price
plt.figure(figsize=(8, 5))

sns.barplot(
    x="Bedrooms",
    y="Price",
    data=df
)

plt.title("Bedrooms vs Average Price")
plt.xlabel("Bedrooms")
plt.ylabel("Average Price")

plt.tight_layout()
plt.show()


# 4. Price Box Plot
plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Price"]
)

plt.title("House Price Box Plot")

plt.tight_layout()
plt.show()


# 5. Correlation Heatmap
plt.figure(figsize=(10, 7))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# EDA Insights
print("\nEDA INSIGHTS")
print("-" * 60)

print("1. Larger houses generally have higher prices.")
print("2. Area shows a strong relationship with house price.")
print("3. Houses with more bedrooms generally have higher prices.")
print("4. Older houses may have lower prices.")
print("5. Distance from the city can negatively affect house prices.")


# Task 4 - Prepare ML Data
print("\n" + "=" * 60)
print("PREPARING MACHINE LEARNING DATA")
print("=" * 60)

features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Age",
    "Parking",
    "Distance"
]

X = df[features]

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))


# Task 5 - Train Linear Regression Model
print("\n" + "=" * 60)
print("TRAINING LINEAR REGRESSION MODEL")
print("=" * 60)

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully.")


# Display coefficients
print("\nMODEL COEFFICIENTS")
print("-" * 60)

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature:12}: {coefficient:,.2f}")

print(f"\nIntercept: {model.intercept_:,.2f}")


# Task 6 - Make Predictions
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results["Error"] = abs(
    results["Actual Price"] -
    results["Predicted Price"]
)

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED PRICES")
print("=" * 60)

print(results.head(10))


# Actual vs Predicted Graph
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()


# Residual Plot
residuals = y_test.values - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Error Plot")

plt.tight_layout()
plt.show()


# Task 7 - Model Evaluation
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

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# Task 8 - Error Analysis
print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

largest_errors = results.nlargest(
    5,
    "Error"
)

smallest_errors = results.nsmallest(
    5,
    "Error"
)

print("\n5 Largest Prediction Errors")
print("-" * 60)
print(largest_errors)

print("\n5 Smallest Prediction Errors")
print("-" * 60)
print(smallest_errors)

print("\nERROR ANALYSIS SUMMARY")
print("-" * 60)

print(
    "The model performs well when the relationship between "
    "house features and price is approximately linear."
)

print(
    "Large prediction errors may occur for unusual or "
    "extreme house records."
)

print(
    "Linear Regression is suitable when the relationship "
    "between the features and price is mostly linear."
)

print(
    "The model could be improved using more real-world data, "
    "feature engineering and advanced regression models."
)


# Task 9 - New House Prediction Function
def predict_house_price(
    area,
    bedrooms,
    bathrooms,
    floors,
    age,
    parking,
    distance
):

    new_house = pd.DataFrame(
        [[
            area,
            bedrooms,
            bathrooms,
            floors,
            age,
            parking,
            distance
        ]],
        columns=features
    )

    prediction = model.predict(new_house)[0]

    return prediction


# Predict prices for 5 new houses
new_houses = pd.DataFrame({
    "Area": [
        1200,
        1500,
        1800,
        2200,
        3000
    ],

    "Bedrooms": [
        2,
        3,
        3,
        4,
        5
    ],

    "Bathrooms": [
        1,
        2,
        2,
        3,
        4
    ],

    "Floors": [
        1,
        1,
        2,
        2,
        3
    ],

    "Age": [
        15,
        10,
        5,
        8,
        2
    ],

    "Parking": [
        1,
        1,
        1,
        2,
        2
    ],

    "Distance": [
        15,
        10,
        8,
        5,
        3
    ]
})

new_houses["Predicted Price"] = model.predict(
    new_houses
)

print("\n" + "=" * 60)
print("5 NEW HOUSE PREDICTIONS")
print("=" * 60)

print(new_houses)


# User Input Prediction
print("\n" + "=" * 60)
print("HOUSE PRICE PREDICTION SYSTEM")
print("=" * 60)

try:

    area = float(
        input("Enter Area (sq.ft): ")
    )

    bedrooms = int(
        input("Enter Bedrooms: ")
    )

    bathrooms = int(
        input("Enter Bathrooms: ")
    )

    floors = int(
        input("Enter Floors: ")
    )

    age = int(
        input("Enter House Age: ")
    )

    parking = int(
        input("Enter Parking Spaces: ")
    )

    distance = float(
        input("Enter Distance from City (km): ")
    )

    estimated_price = predict_house_price(
        area,
        bedrooms,
        bathrooms,
        floors,
        age,
        parking,
        distance
    )

    print(
        f"\nEstimated House Price: "
        f"LKR {estimated_price:,.2f}"
    )

except ValueError:

    print(
        "\nInvalid input. "
        "Please enter numerical values."
    )


print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)