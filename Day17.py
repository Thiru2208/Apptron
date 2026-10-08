import pandas as pd
import numpy as np

# --------------------------------------------
# 1. Create Patient Dataset
# --------------------------------------------

np.random.seed(10)

patient_data = pd.DataFrame({
    "Patient_ID": np.arange(1, 501),
    "Age": np.random.randint(18, 81, 500).astype(float),
    "Gender": np.random.choice(["Male", "Female"], 500),
    "Blood_Pressure": np.random.randint(90, 181, 500).astype(float),
    "Glucose": np.random.randint(70, 201, 500).astype(float),
    "BMI": np.round(np.random.uniform(18, 40, 500), 2),
    "Cholesterol": np.random.randint(120, 301, 500).astype(float)
})


# --------------------------------------------
# 2. Introduce Missing Values
# --------------------------------------------

patient_data.loc[
    np.random.choice(patient_data.index, 20, replace=False),
    "Age"
] = np.nan

patient_data.loc[
    np.random.choice(patient_data.index, 15, replace=False),
    "Gender"
] = np.nan

patient_data.loc[
    np.random.choice(patient_data.index, 25, replace=False),
    "Blood_Pressure"
] = np.nan

patient_data.loc[
    np.random.choice(patient_data.index, 20, replace=False),
    "Glucose"
] = np.nan

patient_data.loc[
    np.random.choice(patient_data.index, 15, replace=False),
    "BMI"
] = np.nan

patient_data.loc[
    np.random.choice(patient_data.index, 20, replace=False),
    "Cholesterol"
] = np.nan


# --------------------------------------------
# 3. Display Dataset Before Cleaning
# --------------------------------------------

print("\nPATIENT DATASET BEFORE CLEANING")
print("----------------------------------------")
print(patient_data.head(10))


# --------------------------------------------
# 4. Detect Missing Values
# --------------------------------------------

print("\nMISSING VALUES")
print("----------------------------------------")

missing_values_before = patient_data.isnull().sum()

print(missing_values_before)


# --------------------------------------------
# 5. Display Rows With Missing Values
# --------------------------------------------

print("\nROWS CONTAINING MISSING VALUES")
print("----------------------------------------")

missing_rows = patient_data[
    patient_data.isnull().any(axis=1)
]

print(missing_rows.head(10))


# --------------------------------------------
# 6. Calculate Missing Percentage
# --------------------------------------------

print("\nMISSING VALUE PERCENTAGE")
print("----------------------------------------")

missing_percentage = (
    patient_data.isnull().sum()
    / len(patient_data)
) * 100

print(missing_percentage.round(2))


# --------------------------------------------
# 7. Store Information Before Cleaning
# --------------------------------------------

total_records_before = len(patient_data)

total_missing_before = patient_data.isnull().sum().sum()


# --------------------------------------------
# 8. Fill Numerical Missing Values
# --------------------------------------------

numerical_columns = [
    "Age",
    "Blood_Pressure",
    "Glucose",
    "BMI",
    "Cholesterol"
]

for column in numerical_columns:

    median_value = patient_data[column].median()

    patient_data[column] = patient_data[column].fillna(
        median_value
    )


# --------------------------------------------
# 9. Fill Categorical Missing Values
# --------------------------------------------

gender_mode = patient_data["Gender"].mode()[0]

patient_data["Gender"] = patient_data["Gender"].fillna(
    gender_mode
)


# --------------------------------------------
# 10. Missing Values After Cleaning
# --------------------------------------------

print("\nMISSING VALUES AFTER CLEANING")
print("----------------------------------------")

missing_values_after = patient_data.isnull().sum()

print(missing_values_after)


# --------------------------------------------
# 11. Display Cleaned Dataset
# --------------------------------------------

print("\nPATIENT DATASET AFTER CLEANING")
print("----------------------------------------")

print(patient_data.head(10))


# --------------------------------------------
# 12. Compare Before and After Cleaning
# --------------------------------------------

comparison = pd.DataFrame({
    "Missing_Before": missing_values_before,
    "Missing_After": missing_values_after
})

print("\nBEFORE AND AFTER CLEANING COMPARISON")
print("----------------------------------------")

print(comparison)


# --------------------------------------------
# 13. Final Cleaning Report
# --------------------------------------------

total_records_after = len(patient_data)

total_missing_after = patient_data.isnull().sum().sum()

rows_removed = total_records_before - total_records_after


if total_missing_after == 0:
    cleaning_status = "Cleaning Completed Successfully"
else:
    cleaning_status = "Cleaning Incomplete"


print("\nPATIENT DATA CLEANING REPORT")
print("----------------------------------------")

print("Total Records:", total_records_after)

print(
    "Missing Values Before:",
    total_missing_before
)

print(
    "Missing Values After:",
    total_missing_after
)

print(
    "Rows Removed:",
    rows_removed
)

print(
    "Cleaning Status:",
    cleaning_status
)