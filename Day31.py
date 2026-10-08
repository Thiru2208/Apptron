import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("=" * 60)
print("HOUSE PRICE DATASET PREPARATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. CREATE DATASET IF FILE DOES NOT EXIST
# ------------------------------------------------------------

file_name = "house_data.csv"

if not os.path.exists(file_name):

    data = {
        "Area": [
            1200, 1500, 1800, 2000, 1100,
            1700, 2200, 1400, 1900, 2500,
            1300, 1600, 2100, 2300, 1250,
            1750, 2400, 1550, 1950, 2600
        ],

        "Bedrooms": [
            2, 3, 3, 4, 2,
            3, 4, 2, 3, 5,
            2, 3, 4, 4, 2,
            3, 4, 3, 3, 5
        ],

        "Bathrooms": [
            1, 2, 2, 3, 1,
            2, 3, 1, 2, 3,
            1, 2, 2, 3, 1,
            2, 3, 2, 2, 4
        ],

        "Age": [
            15, 10, 8, 5, 20,
            7, 4, 12, 6, 3,
            18, 9, 5, 4, 14,
            7, 3, 10, 6, 2
        ],

        "Price": [
            18000000, 24000000, 29000000, 35000000, 16000000,
            27000000, 39000000, 21000000, 31000000, 45000000,
            19000000, 25000000, 36000000, 41000000, 17500000,
            28000000, 43000000, 23500000, 32000000, 48000000
        ]
    }

    df = pd.DataFrame(data)

    df.to_csv(file_name, index=False)

    print("\nhouse_data.csv created successfully!")

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(file_name)

print("\nDATASET LOADED SUCCESSFULLY")

# ------------------------------------------------------------
# 3. DISPLAY DATASET
# ------------------------------------------------------------

print("\nFIRST 5 RECORDS")
print("-" * 60)
print(df.head())

print("\nDATASET SHAPE")
print("-" * 60)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# ------------------------------------------------------------
# 4. FEATURES AND TARGET
# ------------------------------------------------------------

feature_columns = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age"
]

target_column = "Price"

X = df[feature_columns]

y = df[target_column]

print("\nFEATURES (X)")
print("-" * 60)
print(X.head())

print("\nTARGET (y)")
print("-" * 60)
print(y.head())

# ------------------------------------------------------------
# 5. 80% TRAIN / 20% TEST
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("80% TRAIN / 20% TEST SPLIT")
print("=" * 60)

print("Total Records      :", len(df))
print("Training Records   :", len(X_train))
print("Testing Records    :", len(X_test))
print("Number of Features :", X.shape[1])
print("Target Column      :", target_column)

# ------------------------------------------------------------
# 6. 70% TRAIN / 30% TEST
# ------------------------------------------------------------

X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\n" + "=" * 60)
print("70% TRAIN / 30% TEST SPLIT")
print("=" * 60)

print("Total Records      :", len(df))
print("Training Records   :", len(X_train_70))
print("Testing Records    :", len(X_test_30))
print("Number of Features :", X.shape[1])
print("Target Column      :", target_column)

# ------------------------------------------------------------
# 7. COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT COMPARISON")
print("=" * 60)

print(f"{'Split':<15}{'Training':<15}{'Testing':<15}")
print("-" * 45)

print(
    f"{'80 / 20':<15}"
    f"{len(X_train):<15}"
    f"{len(X_test):<15}"
)

print(
    f"{'70 / 30':<15}"
    f"{len(X_train_70):<15}"
    f"{len(X_test_30):<15}"
)

print("\nDataset preparation completed successfully!")