import pandas as pd

# 1. Create Employee DataFrame

employee_data = {
    "Employee_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Name": [
        "Arun", "Banu", "Charan", "Divya", "Eshan",
        "Fathima", "Gokul", "Hema", "Ishan", "Janu"
    ],
    "Age": [25, 28, 30, 26, 35, 29, 32, 27, 31, 24],
    "Department": [
        "IT", "HR", "Finance", "Sales", "IT",
        "Marketing", "Finance", "HR", "Sales", "IT"
    ],
    "Salary": [
        50000, 45000, 60000, 48000, 75000,
        52000, 65000, 47000, 55000, 43000
    ],
    "Experience": [2, 4, 6, 3, 10, 5, 7, 3, 6, 1],
    "Performance_Score": [85, 78, 92, 88, 95, 81, 90, 76, 89, 80]
}

df = pd.DataFrame(employee_data)


# 2. Display Complete Dataset

print("COMPLETE EMPLOYEE DATASET")
print("-------------------------")
print(df)


# 3. Display Selected Columns

print("\nSELECTED COLUMNS")
print("----------------")
print(df[["Name", "Department", "Salary"]])


# 4. Display First 5 Employees

print("\nFIRST 5 EMPLOYEES")
print("-----------------")
print(df.head())


# 5. Display Last 5 Employees

print("\nLAST 5 EMPLOYEES")
print("----------------")
print(df.tail())


# 6. Add Bonus Column
# Bonus = 10% of Salary

df["Bonus"] = df["Salary"] * 0.10

print("\nDATASET AFTER ADDING BONUS")
print("--------------------------")
print(df)


# 7. Rename Salary to Base_Salary

df.rename(columns={"Salary": "Base_Salary"}, inplace=True)

print("\nAFTER RENAMING SALARY TO BASE_SALARY")
print("------------------------------------")
print(df)


# 8. Remove One Unnecessary Column
# Here we remove Age column

df.drop(columns=["Age"], inplace=True)

print("\nAFTER REMOVING AGE COLUMN")
print("-------------------------")
print(df)


# 9. Select One Employee Using loc
# loc uses label/index

print("\nSELECT EMPLOYEE USING loc")
print("-------------------------")
print(df.loc[2])


# 10. Select One Employee Using iloc
# iloc uses position

print("\nSELECT EMPLOYEE USING iloc")
print("--------------------------")
print(df.iloc[4])


# 11. Create Total_Salary Column

df["Total_Salary"] = df["Base_Salary"] + df["Bonus"]

print("\nFINAL DATASET WITH TOTAL SALARY")
print("-------------------------------")
print(df)