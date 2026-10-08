import pandas as pd
import numpy as np
 # 1. CREATE HEALTHCARE DATASET 
data = {
    "Patient_ID": [101, 102, 103, 104, 105, 106, 107, 108, 108, 110],

    "Age": [25, 34, -5, 45, 200, 29, 38, np.nan, 41, "Thirty"],

    "Gender": [
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "female",
        "MALE",
        np.nan,
        "Female",
        "Male"
    ],

    "Blood_Pressure": [
        120,
        130,
        115,
        np.nan,
        140,
        125,
        135,
        128,
        128,
        150
    ],

    "Glucose": [
        90,
        110,
        95,
        180,
        np.nan,
        105,
        120,
        115,
        115,
        200
    ],

    "Weight": [
        65,
        72,
        55,
        80,
        90,
        np.nan,
        68,
        75,
        75,
        85
    ],

    "Appointment_Date": [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
        "wrong-date",
        "2026-08-05",
        "2026-08-06",
        "2026-08-07",
        "2026-08-08",
        "2026-08-08",
        "2026-08-10"
    ]
}

df = pd.DataFrame(data)

print("HEALTHCARE DATASET")
print("------------------------------")
print(df)

 # 2. BASIC DATA INSPECTION 
print("\nFIRST 5 RECORDS")
print("------------------------------")
print(df.head())


print("\nLAST 5 RECORDS")
print("------------------------------")
print(df.tail())


print("\nDATASET SHAPE")
print("------------------------------")
print(df.shape)


print("\nCOLUMN NAMES")
print("------------------------------")
print(df.columns)


print("\nDATA TYPES")
print("------------------------------")
print(df.dtypes)


print("\nDATASET INFORMATION")
print("------------------------------")
df.info()


print("\nSTATISTICAL SUMMARY")
print("------------------------------")
print(df.describe(include="all"))

 # 3. CHECK MISSING VALUES 
print("\nMISSING VALUES")
print("------------------------------")

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

 # 4. CHECK DUPLICATE RECORDS 
print("\nDUPLICATE RECORDS")
print("------------------------------")

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


print("\nDuplicate Rows:")
print(df[df.duplicated()])

 # 5. CHECK DUPLICATE PATIENT IDS 
print("\nDUPLICATE PATIENT IDs")
print("------------------------------")

duplicate_patients = df[
    df.duplicated(subset="Patient_ID", keep=False)
]

print(duplicate_patients)

print(
    "Duplicate Patient ID Count:",
    df["Patient_ID"].duplicated().sum()
)

 # 6. CHECK WRONG AGE DATA TYPES 
print("\nAGE DATA TYPE PROBLEM")
print("------------------------------")

print("Age column data type before conversion:")
print(df["Age"].dtype)


# Convert Age into numeric
# Invalid text such as "Thirty" becomes NaN

df["Age_Numeric"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

print("\nAge after converting to numeric:")
print(df["Age_Numeric"])

 # 7. FIND INVALID AGE RECORDS 
print("\nINVALID AGE RECORDS")
print("------------------------------")

invalid_age = df[
    (df["Age_Numeric"] < 0)
    |
    (df["Age_Numeric"] > 120)
]

print(invalid_age)

 # 8. CHECK APPOINTMENT DATE DATA TYPE 
print("\nAPPOINTMENT DATE CHECK")
print("------------------------------")

print("Before conversion:")
print(df["Appointment_Date"].dtype)


# Convert appointment date to datetime
# Invalid dates become NaT

df["Appointment_Date_Clean"] = pd.to_datetime(
    df["Appointment_Date"],
    errors="coerce"
)

print("\nAfter conversion:")
print(df["Appointment_Date_Clean"])

 # 9. INVALID APPOINTMENT DATES 
print("\nINVALID APPOINTMENT DATES")
print("------------------------------")

invalid_dates = df[
    df["Appointment_Date_Clean"].isna()
]

print(invalid_dates)

 # 10. CHECK GENDER TEXT CONSISTENCY 
print("\nGENDER VALUES BEFORE CLEANING")
print("------------------------------")

print(df["Gender"].value_counts(dropna=False))


# Standardize gender values

df["Gender_Clean"] = (
    df["Gender"]
    .astype("string")
    .str.strip()
    .str.title()
)

print("\nGENDER VALUES AFTER CLEANING")
print("------------------------------")

print(df["Gender_Clean"].value_counts(dropna=False))

 # 11. FIND INVALID RECORDS 
print("\nINVALID RECORDS")
print("------------------------------")

invalid_records = df[
    (df["Age_Numeric"] < 0)
    |
    (df["Age_Numeric"] > 120)
    |
    (df["Age_Numeric"].isna())
    |
    (df["Appointment_Date_Clean"].isna())
    |
    (df["Blood_Pressure"].isna())
    |
    (df["Glucose"].isna())
    |
    (df["Weight"].isna())
    |
    (df["Gender"].isna())
]

print(invalid_records)

 # 12. AUTOMATIC DATA QUALITY REPORT 
print("\n")
print("=" * 50)
print("        DATA QUALITY REPORT")
print("=" * 50)


print("\n1. Dataset Shape")
print("------------------------------")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


print("\n2. Data Types")
print("------------------------------")
print(df.dtypes)


print("\n3. Missing Values")
print("------------------------------")
print(df.isnull().sum())


print("\n4. Duplicate Rows")
print("------------------------------")
print("Duplicate rows:", df.duplicated().sum())


print("\n5. Duplicate Patient IDs")
print("------------------------------")
print(
    "Duplicate Patient IDs:",
    df["Patient_ID"].duplicated().sum()
)


print("\n6. Invalid Ages")
print("------------------------------")

invalid_age_count = (
    (df["Age_Numeric"] < 0)
    |
    (df["Age_Numeric"] > 120)
    |
    (df["Age_Numeric"].isna())
).sum()

print("Invalid Age Count:", invalid_age_count)


print("\n7. Invalid Appointment Dates")
print("------------------------------")

invalid_date_count = (
    df["Appointment_Date_Clean"]
    .isna()
    .sum()
)

print(
    "Invalid Appointment Date Count:",
    invalid_date_count
)


print("\n8. Total Invalid Records")
print("------------------------------")

print(
    "Invalid Record Count:",
    len(invalid_records)
)

 # 13. CALCULATE DATA QUALITY SCORE 
total_cells = (
    df.shape[0]
    *
    df.shape[1]
)

missing_cells = (
    df.isnull()
    .sum()
    .sum()
)

quality_score = (
    (total_cells - missing_cells)
    /
    total_cells
) * 100


print("\n9. Data Quality Score")
print("------------------------------")

print(
    f"Data Quality Score: {quality_score:.2f}%"
)


print("\n")
print("=" * 50)
print("        END OF AUDIT REPORT")
print("=" * 50)