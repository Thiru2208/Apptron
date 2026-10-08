import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# HOUSE PRICE RELATIONSHIP ANALYZER
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# 1. CREATE 100 HOUSE RECORDS
# ------------------------------------------------------------

house_id = np.arange(1, 101)

area = np.random.randint(500, 3001, 100)

bedrooms = np.random.randint(1, 6, 100)

bathrooms = np.random.randint(1, 4, 100)

age = np.random.randint(0, 41, 100)

# Create price with a positive relationship to area
# Price mainly depends on house area
noise = np.random.normal(0, 15000, 100)

price = (
    area * 100
    + bedrooms * 10000
    + bathrooms * 8000
    - age * 1000
    + noise
)

# Make sure prices are positive
price = np.maximum(price, 50000)

# ------------------------------------------------------------
# 2. ADD SOME OUTLIERS
# ------------------------------------------------------------

price[10] = price[10] + 180000
price[45] = price[45] + 220000
price[80] = price[80] - 100000

# ------------------------------------------------------------
# 3. CREATE DATAFRAME
# ------------------------------------------------------------

data = pd.DataFrame({
    "House_ID": house_id,
    "Area_sqft": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age": age,
    "Price": np.round(price, 2)
})

# ------------------------------------------------------------
# 4. SAVE DATASET
# ------------------------------------------------------------

data.to_csv("house_price_dataset.csv", index=False)

print("=" * 65)
print("HOUSE PRICE RELATIONSHIP ANALYZER")
print("=" * 65)

print("\nDataset created successfully!")
print("Total records:", len(data))

print("\nFIRST 10 HOUSE RECORDS")
print("-" * 65)
print(data.head(10))

# ------------------------------------------------------------
# 5. LOAD DATASET USING PANDAS
# ------------------------------------------------------------

df = pd.read_csv("house_price_dataset.csv")

print("\nDATASET INFORMATION")
print("-" * 65)
print(df.info())

print("\nSTATISTICAL SUMMARY")
print("-" * 65)
print(df.describe())

# ------------------------------------------------------------
# 6. CALCULATE CORRELATION
# ------------------------------------------------------------

correlation = df["Area_sqft"].corr(df["Price"])

print("\nAREA VS PRICE CORRELATION")
print("-" * 65)
print(f"Correlation: {correlation:.3f}")

if correlation > 0.7:
    relationship = "Strong Positive Relationship"
elif correlation > 0.3:
    relationship = "Moderate Positive Relationship"
elif correlation > 0:
    relationship = "Weak Positive Relationship"
elif correlation < -0.7:
    relationship = "Strong Negative Relationship"
elif correlation < -0.3:
    relationship = "Moderate Negative Relationship"
elif correlation < 0:
    relationship = "Weak Negative Relationship"
else:
    relationship = "No Relationship"

print("Relationship:", relationship)

# ------------------------------------------------------------
# 7. IDENTIFY POSSIBLE OUTLIERS USING IQR
# ------------------------------------------------------------

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Price"] < lower_bound) |
    (df["Price"] > upper_bound)
]

print("\nPOSSIBLE PRICE OUTLIERS")
print("-" * 65)

if len(outliers) > 0:
    print(outliers[
        ["House_ID", "Area_sqft", "Price"]
    ])
else:
    print("No possible outliers detected.")

# ------------------------------------------------------------
# 8. AREA VS PRICE SCATTER PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Area_sqft"],
    df["Price"],
    alpha=0.7
)

plt.title("House Area vs House Price")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. APPROXIMATE REGRESSION LINE - BONUS
# ------------------------------------------------------------

# Find approximate best-fit line using NumPy
slope, intercept = np.polyfit(
    df["Area_sqft"],
    df["Price"],
    1
)

predicted_price = (
    slope * df["Area_sqft"] + intercept
)

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Area_sqft"],
    df["Price"],
    alpha=0.7,
    label="Actual Houses"
)

# Sort values so the regression line appears properly
sorted_area = np.sort(df["Area_sqft"])

regression_line = (
    slope * sorted_area + intercept
)

plt.plot(
    sorted_area,
    regression_line,
    linewidth=2,
    label="Approximate Regression Line"
)

plt.title("Area vs Price with Approximate Regression Line")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. SHORT REPORT
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("SHORT REPORT")
print("=" * 65)

print("Feature        : Area_sqft")
print("Target         : Price")
print(f"Correlation    : {correlation:.3f}")
print(f"Relationship   : {relationship}")

print(f"Possible Outliers : {len(outliers)}")

if correlation > 0:
    print("Positive Relationship : Yes")
    print("Negative Relationship : No")
else:
    print("Positive Relationship : No")
    print("Negative Relationship : Yes")

if abs(correlation) >= 0.7:
    suitable = "Yes"
    reason = (
        "Area and price show a strong linear relationship, "
        "so Linear Regression is suitable."
    )
else:
    suitable = "Possibly"
    reason = (
        "The relationship is not very strong, therefore other "
        "features may also need to be considered."
    )

print(f"Is Linear Regression Suitable? : {suitable}")
print(f"Reason : {reason}")

print("\nREGRESSION LINE")
print("-" * 65)
print(
    f"Price = {slope:.2f} × Area + {intercept:.2f}"
)

print("\nCONCLUSION")
print("-" * 65)
print(
    "As house area increases, house price generally increases. "
    "The scatter plot shows an approximately linear positive "
    "relationship between Area and Price. Therefore, house area "
    "can be useful for predicting house price using Linear Regression."
)

print("=" * 65)