import pandas as pd
import numpy as np
import os

# Get current Python file folder
folder = os.path.dirname(os.path.abspath(__file__))

# Create dataset
np.random.seed(10)

data = {
    "Order_ID": range(1, 1001),

    "Customer_ID": np.random.randint(100, 500, 1000),

    "Product": np.random.choice(
        ["Laptop", "Phone", "Mouse", "Keyboard", "Monitor"],
        1000
    ),

    "Category": np.random.choice(
        ["Electronics", "Accessories", "Office"],
        1000
    ),

    "Quantity": np.random.randint(1, 10, 1000).astype(float),

    "Price": np.random.randint(
        1000, 100000, 1000
    ).astype(float),

    "City": np.random.choice(
        ["Colombo", "Kandy", "Jaffna", "Galle"],
        1000
    ),

    "Order_Date": pd.date_range(
        "2026-01-01",
        periods=1000
    ).astype(str)
}

df = pd.DataFrame(data)


# ----------------------------
# ADD PROBLEMS
# ----------------------------

# Missing prices
df.loc[5, "Price"] = np.nan
df.loc[20, "Price"] = np.nan

# Invalid quantities
df.loc[10, "Quantity"] = -5
df.loc[30, "Quantity"] = 0

# Inconsistent city names
df.loc[15, "City"] = "colombo"
df.loc[25, "City"] = "JAFFNA"

# Wrong dates
df.loc[40, "Order_Date"] = "wrong-date"
df.loc[50, "Order_Date"] = "2026/05/10"

# Add duplicates
df = pd.concat(
    [df, df.iloc[[1, 2, 3]]],
    ignore_index=True
)


# ----------------------------
# SAVE RAW DATASET
# ----------------------------

raw_file = os.path.join(folder, "raw_orders.csv")

df.to_csv(raw_file, index=False)

print("\nRAW DATASET")
print("-" * 50)

print(df.head(20))

print("\nTotal Raw Records:", len(df))


# ----------------------------
# DATA CLEANING
# ----------------------------

# 1. Remove duplicates
df = df.drop_duplicates()


# 2. Fill missing prices with mean
mean_price = df["Price"].mean()

df["Price"] = df["Price"].fillna(mean_price)


# 3. Fix invalid quantities
df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan

mean_quantity = df["Quantity"].mean()

df["Quantity"] = df["Quantity"].fillna(mean_quantity)


# 4. Standardize city names
df["City"] = df["City"].str.title()


# 5. Convert dates
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    format="mixed",
    errors="coerce"
)


# 6. Remove invalid dates
df = df.dropna(subset=["Order_Date"])


# 7. Convert quantity to integer
df["Quantity"] = df["Quantity"].astype(int)


# 8. Create Total Amount
df["Total_Amount"] = (
    df["Quantity"] * df["Price"]
)


# ----------------------------
# SAVE CLEAN DATASET
# ----------------------------

clean_file = os.path.join(folder, "clean_orders.csv")

df.to_csv(clean_file, index=False)


# ----------------------------
# FINAL OUTPUT
# ----------------------------

print("\nCLEAN DATASET")
print("-" * 50)

print(df.head(20))


print("\nMissing Values")
print("-" * 50)

print(df.isnull().sum())


print("\nTotal Clean Records:", len(df))

print("\nCleaning Completed Successfully!")

print("\nFiles Saved:")
print(raw_file)
print(clean_file)