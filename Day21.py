import pandas as pd

# Load sales data
df = pd.read_csv(r"C:\Users\LENOVO\Apptron\sales_data.csv")

print("\nSALES DATASET")
print("-" * 80)
print(df)


# Dataset  information

print("\nDATASET INFORMATION")
print("=" * 80)

print("Number of Rows    :", len(df))
print("Number of Columns :", len(df.columns))

print("\nColumn Names:")
print(df.columns.tolist())

# Check for missing values

print("\nMISSING VALUES")
print("=" * 80) 

print(df.isnull().sum())

# Verify revenue
# Revenue = Quantity * Price

df["Calculated_Revenue"] = df["Quantity"] * df["Price"]

revenue_check = (
    df["Revenue"] == df["Calculated_Revenue"]
).all()

print("\n" + "=" * 80)
print("REVENUE VERIFICATION")
print("=" * 80)

if revenue_check:
    print("All Revenue values are correct.")
else:
    print("Some Revenue values are incorrect.")


# Remove temporary calculation column
df.drop(
    columns=["Calculated_Revenue"],
    inplace=True
)

# Regional report

city_report = df.groupby("City").agg(

    Total_Revenue=("Revenue", "sum"),

    Average_Revenue=("Revenue", "mean"),

    Maximum_Revenue=("Revenue", "max"),

    Minimum_Revenue=("Revenue", "min"),

    Number_of_Orders=("Sale_ID", "count"),

    Total_Quantity=("Quantity", "sum")

).reset_index()


print("\n" + "=" * 80)
print("CITY / REGIONAL REPORT")
print("=" * 80)

print(city_report)

# Product report

product_report = df.groupby("Product").agg(

    Total_Quantity=("Quantity", "sum"),

    Total_Revenue=("Revenue", "sum"),

    Average_Revenue=("Revenue", "mean"),

    Maximum_Revenue=("Revenue", "max"),

    Number_of_Orders=("Sale_ID", "count")

).reset_index()


print("\n" + "=" * 80)
print("PRODUCT REPORT")
print("=" * 80)

print(product_report)

# Category report

category_report = df.groupby("Category").agg(

    Total_Quantity=("Quantity", "sum"),

    Total_Revenue=("Revenue", "sum"),

    Average_Revenue=("Revenue", "mean"),

    Maximum_Revenue=("Revenue", "max"),

    Number_of_Orders=("Sale_ID", "count")

).reset_index()


print("\n" + "=" * 80)
print("CATEGORY REPORT")
print("=" * 80)

print(category_report)

# Find top city
top_city = city_report.loc[
    city_report["Total_Revenue"].idxmax()
]


# find top product

top_product = product_report.loc[
    product_report["Total_Revenue"].idxmax()
]

# Find top category
top_category = category_report.loc[
    category_report["Total_Revenue"].idxmax()
]

# Overall Business calculation

total_revenue = df["Revenue"].sum()

average_order_value = df["Revenue"].mean()

maximum_order_value = df["Revenue"].max()

minimum_order_value = df["Revenue"].min()

total_quantity = df["Quantity"].sum()

total_orders = df["Sale_ID"].count()

# Final business intelligence report

print("\n" + "=" * 80)
print("BUSINESS INTELLIGENCE REPORT")
print("=" * 80)

print(
    f"Top City             : "
    f"{top_city['City']}"
)

print(
    f"Top Product          : "
    f"{top_product['Product']}"
)

print(
    f"Top Category         : "
    f"{top_category['Category']}"
)

print(
    f"Total Orders         : "
    f"{total_orders}"
)

print(
    f"Total Quantity Sold  : "
    f"{total_quantity}"
)

print(
    f"Total Revenue        : "
    f"{total_revenue:,.2f}"
)

print(
    f"Average Order Value  : "
    f"{average_order_value:,.2f}"
)

print(
    f"Maximum Order Value  : "
    f"{maximum_order_value:,.2f}"
)

print(
    f"Minimum Order Value  : "
    f"{minimum_order_value:,.2f}"
)

print("=" * 80)

# Create final report DataFrame

final_report = pd.DataFrame({

    "Metric": [

        "Top City",

        "Top Product",

        "Top Category",

        "Total Orders",

        "Total Quantity Sold",

        "Total Revenue",

        "Average Order Value",

        "Maximum Order Value",

        "Minimum Order Value"

    ],

    "Result": [

        top_city["City"],

        top_product["Product"],

        top_category["Category"],

        total_orders,

        total_quantity,

        round(total_revenue, 2),

        round(average_order_value, 2),

        round(maximum_order_value, 2),

        round(minimum_order_value, 2)

    ]

})


print("\n" + "=" * 80)
print("FINAL STRUCTURED REPORT")
print("=" * 80)

print(final_report)

# Export reports to csv

city_report.to_csv(
    r"C:\Users\LENOVO\Apptron\city_report.csv",
    index=False
)

product_report.to_csv(
    r"C:\Users\LENOVO\Apptron\product_report.csv",
    index=False
)

category_report.to_csv(
    r"C:\Users\LENOVO\Apptron\category_report.csv",
    index=False
)

final_report.to_csv(
    r"C:\Users\LENOVO\Apptron\final_business_report.csv",
    index=False
)

# confirm export

print("\n" + "=" * 80)
print("REPORTS EXPORTED SUCCESSFULLY")
print("=" * 80)

print(r"1. C:\Users\LENOVO\Apptron\city_report.csv")
print(r"2. C:\Users\LENOVO\Apptron\product_report.csv")
print(r"3. C:\Users\LENOVO\Apptron\category_report.csv")
print(r"4. C:\Users\LENOVO\Apptron\final_business_report.csv")