import pandas as pd
import os

filename = input("Enter the filename (with .csv extension): ")

file_path = os.path.join(
    r"C:\Users\LENOVO\Apptron",
    filename
)

df = pd.read_csv(file_path)

print("Sales dataset")
print("-------------")
print(df)   

# Dataset Informationsales_data.csv

print("\nDATASET INFORMATION")
print("--------------------------------")

print("Number of rows:", df.shape[0])

print("Number of columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# Display first records

print("\nFIRST 5 RECORDS")
print("--------------------------------")

print(df.head())


# Display last records

print("\nLAST 5 RECORDS")
print("--------------------------------")

print(df.tail())


# Calculate total revenue

total_revenue = df["Revenue"].sum()

print("\nTOTAL REVENUE")
print("--------------------------------")

print("Total Revenue:", total_revenue)


# Find highest-value order

highest_order = df.loc[df["Revenue"].idxmax()]

print("\nHIGHEST VALUE ORDER")
print("--------------------------------")

print(highest_order)


output_file = r"C:\Users\LENOVO\Apptron\analyzed_sales_data.csv"

df.to_csv(output_file, index=False)

print("\nAnalyzed data saved successfully!")
print("File Name:", output_file)