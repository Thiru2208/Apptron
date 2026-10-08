import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("house_price_dataset.csv")

print("=" * 60)
print("HOUSE PRICE PREDICTION - LINEAR REGRESSION")
print("=" * 60)

print("\nFIRST 10 RECORDS")
print("-" * 60)
print(df.head(10))


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

X = df[["Area_sqft", "Bedrooms", "Bathrooms", "Age"]]

y = df["Price"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDATASET SPLIT")
print("-" * 60)
print("Training records :", len(X_train))
print("Testing records  :", len(X_test))


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 5. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nMODEL TRAINING")
print("-" * 60)
print("Linear Regression model trained successfully!")


# ============================================================
# 6. PREDICT TEST DATA
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 7. ACTUAL VS PREDICTED
# ============================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

comparison["Actual Price"] = comparison["Actual Price"].round(2)
comparison["Predicted Price"] = comparison["Predicted Price"].round(2)

print("\nACTUAL PRICE VS PREDICTED PRICE")
print("-" * 60)
print(comparison)


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-" * 60)
print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"R² Score            : {r2:.4f}")


# ============================================================
# 9. MODEL PARAMETERS
# ============================================================

print("\nMODEL PARAMETERS")
print("-" * 60)

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature:<12}: {coefficient:.2f}")

print(f"Intercept   : {model.intercept_:.2f}")


# ============================================================
# 10. PREDICT 5 NEW HOUSES
# ============================================================

new_houses = pd.DataFrame({
    "Area_sqft":  [1800, 1500, 2200, 1200, 2500],
    "Bedrooms":   [3,    2,    4,    2,    5],
    "Bathrooms":  [2,    1,    3,    2,    4],
    "Age":        [5,    10,   3,    15,   2]
})

new_predictions = model.predict(new_houses)

new_houses["Predicted_Price"] = new_predictions.round(2)

print("\nPREDICTIONS FOR 5 NEW HOUSES")
print("-" * 60)
print(new_houses)


# ============================================================
# 11. EXPLANATION
# ============================================================

print("\nSHORT EXPLANATION")
print("-" * 60)

print(
    "The Linear Regression model learned the relationship between "
    "Area_sqft, Bedrooms, Bathrooms and Age with the house Price. "
    "After training, the model predicted prices for the test data "
    "and five new houses."
)

print("\n" + "=" * 60)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 60)