import pandas as pd
import matplotlib.pyplot as plt

 # 1. CREATE SALES DATASET 

data = {
    "Month": [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ],

    "Revenue": [
        45000, 52000, 48000, 61000,
        58000, 72000, 68000, 75000,
        70000, 82000, 90000, 95000
    ],

    "Orders": [
        120, 135, 128, 150,
        145, 175, 165, 180,
        170, 195, 210, 220
    ],

    "Profit": [
        12000, 15000, 13500, 18000,
        16500, 22000, 20000, 24000,
        21500, 26000, 30000, 32000
    ],

    "Product_Category": [
        "Electronics", "Clothing", "Food", "Electronics",
        "Clothing", "Food", "Electronics", "Clothing",
        "Food", "Electronics", "Clothing", "Food"
    ]
}

df = pd.DataFrame(data)

 # 2. DISPLAY DATASET 

print("\nSALES PERFORMANCE DATASET")
print("-" * 70)
print(df)

 # 3. LINE CHART - MONTHLY REVENUE TREND 

plt.figure(figsize=(10, 5))

plt.plot(
    df["Month"],
    df["Revenue"],
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

 # 4. BAR CHART - REVENUE BY CATEGORY 

revenue_by_category = df.groupby(
    "Product_Category"
)["Revenue"].sum()

print("\nREVENUE BY CATEGORY")
print("-" * 40)
print(revenue_by_category)

plt.figure(figsize=(8, 5))

revenue_by_category.plot(kind="bar")

plt.title("Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

 # 5. PIE CHART - ORDERS BY CATEGORY 

orders_by_category = df.groupby(
    "Product_Category"
)["Orders"].sum()

print("\nORDERS BY CATEGORY")
print("-" * 40)
print(orders_by_category)

plt.figure(figsize=(7, 7))

plt.pie(
    orders_by_category,
    labels=orders_by_category.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Orders by Product Category")

plt.show()

 # 6. ANALYSIS 
# Highest revenue month

highest_revenue_index = df["Revenue"].idxmax()

highest_revenue_month = df.loc[
    highest_revenue_index,
    "Month"
]

highest_revenue = df.loc[
    highest_revenue_index,
    "Revenue"
]

# Lowest revenue month
lowest_revenue_index = df["Revenue"].idxmin()

lowest_revenue_month = df.loc[
    lowest_revenue_index,
    "Month"
]

lowest_revenue = df.loc[
    lowest_revenue_index,
    "Revenue"
]

# Profit by category
profit_by_category = df.groupby(
    "Product_Category"
)["Profit"].sum()

most_profitable_category = profit_by_category.idxmax()

highest_profit = profit_by_category.max()

# Category with highest order count
highest_order_category = orders_by_category.idxmax()

highest_order_count = orders_by_category.max()


# Display analysis
print("\n" + "=" * 60)
print("SALES PERFORMANCE ANALYSIS")
print("=" * 60)

print(
    "Highest Revenue Month :",
    highest_revenue_month
)

print(
    "Highest Revenue       :",
    highest_revenue
)

print(
    "\nLowest Revenue Month  :",
    lowest_revenue_month
)

print(
    "Lowest Revenue        :",
    lowest_revenue
)

print(
    "\nMost Profitable Category :",
    most_profitable_category
)

print(
    "Total Profit             :",
    highest_profit
)

print(
    "\nCategory with Highest Order Count :",
    highest_order_category
)

print(
    "Total Orders                       :",
    highest_order_count
)

# 7. REVENUE VS PROFIT COMPARISON 
plt.figure(figsize=(10, 5))

plt.plot(
    df["Month"],
    df["Revenue"],
    marker="o",
    label="Revenue"
)

plt.plot(
    df["Month"],
    df["Profit"],
    marker="s",
    label="Profit"
)

plt.title("Monthly Revenue vs Profit")
plt.xlabel("Month")
plt.ylabel("Amount")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()
plt.show()