import pandas as pd
import numpy as np

 
# SALES PERFORMANCE ANALYZER
# Sorting + Grouping
 

# For getting the same random values every time
np.random.seed(20)

# Number of sales records
number_of_records = 2000
 
# 1. CREATE SALES DATASET

order_id = np.arange(1, number_of_records + 1)

salespersons = [
    "Amal",
    "Kamal",
    "Nimal",
    "Saman",
    "Kasun",
    "Dinesh",
    "Ravi",
    "Tharindu"
]

regions = [
    "Northern",
    "Western",
    "Central",
    "Southern",
    "Eastern"
]

products = [
    "Laptop",
    "Phone",
    "Monitor",
    "Keyboard",
    "Mouse",
    "Printer",
    "Headphones",
    "Tablet"
]

categories = [
    "Electronics",
    "Accessories",
    "Office"
]


# Random data generation

salesperson = np.random.choice(
    salespersons,
    number_of_records
)

region = np.random.choice(
    regions,
    number_of_records
)

product = np.random.choice(
    products,
    number_of_records
)

category = np.random.choice(
    categories,
    number_of_records
)

quantity = np.random.randint(
    1,
    21,
    number_of_records
)

revenue = np.random.randint(
    5000,
    100000,
    number_of_records
)

# Profit is generated as 10% - 35% of revenue
profit_percentage = np.random.uniform(
    0.10,
    0.35,
    number_of_records
)

profit = revenue * profit_percentage

profit = np.round(profit, 2)


 
# CREATE DATAFRAME

sales_data = pd.DataFrame({

    "Order_ID": order_id,

    "Salesperson": salesperson,

    "Region": region,

    "Product": product,

    "Category": category,

    "Quantity": quantity,

    "Revenue": revenue,

    "Profit": profit
})
 
# 2. DISPLAY SALES DATASET

print("\nSALES PERFORMANCE DATASET")
print("-" * 80)

print(sales_data)

print("\nTotal Number of Records:")
print(len(sales_data))

# 3. DATASET INFORMATION

print("\nDATASET INFORMATION")
print("-" * 80)

print("\nShape of Dataset:")
print(sales_data.shape)

print("\nColumn Names:")
print(sales_data.columns)

print("\nData Types:")
print(sales_data.dtypes)

# 4. FIRST 10 RECORDS
 
print("\nFIRST 10 RECORDS")
print("-" * 80)

print(sales_data.head(10))

 
# SORTING DATA 
 
# 5. SORT BY REVENUE - ASCENDING

revenue_ascending = sales_data.sort_values(
    by="Revenue",
    ascending=True
)

print("\nREVENUE - ASCENDING ORDER")
print("-" * 80)

print(
    revenue_ascending[
        [
            "Order_ID",
            "Salesperson",
            "Region",
            "Revenue"
        ]
    ].head(10)
)


# 6. SORT BY REVENUE - DESCENDING

revenue_descending = sales_data.sort_values(
    by="Revenue",
    ascending=False
)

print("\nREVENUE - DESCENDING ORDER")
print("-" * 80)

print(
    revenue_descending[
        [
            "Order_ID",
            "Salesperson",
            "Region",
            "Revenue"
        ]
    ].head(10)
)
 
# 7. SORT BY PROFIT
 

profit_sorted = sales_data.sort_values(
    by="Profit",
    ascending=False
)

print("\nHIGHEST PROFIT SALES")
print("-" * 80)

print(
    profit_sorted[
        [
            "Order_ID",
            "Salesperson",
            "Product",
            "Revenue",
            "Profit"
        ]
    ].head(10)
)

# 8. SORT BY QUANTITY
 

quantity_sorted = sales_data.sort_values(
    by="Quantity",
    ascending=False
)

print("\nHIGHEST QUANTITY SALES")
print("-" * 80)

print(
    quantity_sorted[
        [
            "Order_ID",
            "Product",
            "Quantity"
        ]
    ].head(10)
)
 
# 9. SORT USING MULTIPLE COLUMNS
 

multiple_sort = sales_data.sort_values(
    by=["Region", "Revenue"],
    ascending=[True, False]
)

print("\nSORT BY REGION AND REVENUE")
print("-" * 80)

print(
    multiple_sort[
        [
            "Order_ID",
            "Region",
            "Salesperson",
            "Revenue"
        ]
    ].head(20)
)
 
# 10. SORT BY INDEX
 
index_sorted = sales_data.sort_index(
    ascending=False
)

print("\nSORT BY INDEX - DESCENDING")
print("-" * 80)

print(index_sorted.head(10))

 
# GROUPING DATA 
 
# 11. REVENUE BY REGION
 

revenue_by_region = sales_data.groupby(
    "Region"
)["Revenue"].sum()

revenue_by_region = revenue_by_region.sort_values(
    ascending=False
)

print("\nTOTAL REVENUE BY REGION")
print("-" * 80)

print(revenue_by_region)
 
# 12. REVENUE BY CATEGORY
 
revenue_by_category = sales_data.groupby(
    "Category"
)["Revenue"].sum()

revenue_by_category = revenue_by_category.sort_values(
    ascending=False
)

print("\nTOTAL REVENUE BY CATEGORY")
print("-" * 80)

print(revenue_by_category)

 
# 13. PROFIT BY SALESPERSON
 

profit_by_salesperson = sales_data.groupby(
    "Salesperson"
)["Profit"].sum()

profit_by_salesperson = profit_by_salesperson.sort_values(
    ascending=False
)

print("\nTOTAL PROFIT BY SALESPERSON")
print("-" * 80)

print(profit_by_salesperson)

# 14. TOP SALESPERSON
 

top_salesperson = profit_by_salesperson.idxmax()

top_salesperson_profit = profit_by_salesperson.max()

print("\nTOP SALESPERSON")
print("-" * 80)

print("Salesperson :", top_salesperson)

print(
    "Total Profit :",
    round(top_salesperson_profit, 2)
)

# 15. BEST PERFORMING REGION
 

profit_by_region = sales_data.groupby(
    "Region"
)["Profit"].sum()

profit_by_region = profit_by_region.sort_values(
    ascending=False
)

best_region = profit_by_region.idxmax()

best_region_profit = profit_by_region.max()

print("\nBEST PERFORMING REGION")
print("-" * 80)

print("Region :", best_region)

print(
    "Total Profit :",
    round(best_region_profit, 2)
)

 
# 16. BEST-SELLING CATEGORY

quantity_by_category = sales_data.groupby(
    "Category"
)["Quantity"].sum()

quantity_by_category = quantity_by_category.sort_values(
    ascending=False
)

best_category = quantity_by_category.idxmax()

best_category_quantity = quantity_by_category.max()

print("\nBEST-SELLING CATEGORY")
print("-" * 80)

print("Category :", best_category)

print(
    "Total Quantity Sold :",
    best_category_quantity
)

 
# BONUS 
 
# 17. RANK REGIONS BY TOTAL PROFIT
 
region_ranking = sales_data.groupby(
    "Region"
)["Profit"].sum().reset_index()

region_ranking = region_ranking.sort_values(
    by="Profit",
    ascending=False
)

region_ranking["Rank"] = (
    region_ranking["Profit"]
    .rank(
        method="dense",
        ascending=False
    )
    .astype(int)
)

region_ranking = region_ranking[
    [
        "Rank",
        "Region",
        "Profit"
    ]
]

print("\nREGION RANKING BY TOTAL PROFIT")
print("-" * 80)

print(
    region_ranking.to_string(
        index=False
    )
)


# 18. COMPLETE GROUP SUMMARY

region_summary = sales_data.groupby(
    "Region"
).agg(

    Total_Revenue=("Revenue", "sum"),

    Total_Profit=("Profit", "sum"),

    Total_Quantity=("Quantity", "sum"),

    Number_of_Orders=("Order_ID", "count")

)

region_summary = region_summary.sort_values(
    by="Total_Profit",
    ascending=False
)

print("\nREGION PERFORMANCE SUMMARY")
print("-" * 80)

print(region_summary)


# 19. SALESPERSON PERFORMANCE SUMMARY

salesperson_summary = sales_data.groupby(
    "Salesperson"
).agg(

    Total_Revenue=("Revenue", "sum"),

    Total_Profit=("Profit", "sum"),

    Total_Quantity=("Quantity", "sum"),

    Number_of_Orders=("Order_ID", "count")

)

salesperson_summary = salesperson_summary.sort_values(
    by="Total_Profit",
    ascending=False
)

print("\nSALESPERSON PERFORMANCE SUMMARY")
print("-" * 80)

print(salesperson_summary)


# 20. FINAL RESULTS

print("\n" + "=" * 80)

print("FINAL SALES PERFORMANCE RESULTS")

print("=" * 80)

print(
    "\nTop Salesperson:",
    top_salesperson
)

print(
    "Top Salesperson Profit:",
    round(top_salesperson_profit, 2)
)

print(
    "\nBest Performing Region:",
    best_region
)

print(
    "Best Region Profit:",
    round(best_region_profit, 2)
)

print(
    "\nBest-Selling Category:",
    best_category
)

print(
    "Total Quantity Sold:",
    best_category_quantity
)

print("\nSales Performance Analysis Completed Successfully!")