import numpy as np

# Product Dataset
# Columns: Product ID, Monthly Sales, Revenue, Profit
data = np.array([
    [1, 120, 60000, 12000],
    [2, 85, 42500, 8500],
    [3, 150, 75000, 18000],
    [4, 60, 30000, 5000],
    [5, 200, 100000, 25000],
    [6, 110, 55000, 11000],
    [7, 95, 47500, 9000],
    [8, 175, 87500, 21000],
    [9, 70, 35000, 6000],
    [10, 140, 70000, 15000]
])

product_id = data[:, 0]
sales = data[:, 1]
revenue = data[:, 2]
profit = data[:, 3]

print("SALES ANALYTICS DASHBOARD")
print("-------------------------")

# Total Revenue
total_revenue = np.sum(revenue)
print("Total Revenue:", total_revenue)

# Average Revenue
average_revenue = np.mean(revenue)
print("Average Revenue:", average_revenue)

# Highest Sales
highest_sales = np.max(sales)
print("Highest Sales:", highest_sales)

# Lowest Sales
lowest_sales = np.min(sales)
print("Lowest Sales:", lowest_sales)

# Profit Percentage
profit_percentage = (profit / revenue) * 100

print("\nPROFIT PERCENTAGE")
for i in range(len(product_id)):
    print(
        "Product", product_id[i],
        ":", round(profit_percentage[i], 2), "%"
    )

# Best Selling Product
best_index = np.argmax(sales)

print("\nBest Selling Product:")
print("Product ID:", product_id[best_index])
print("Sales:", sales[best_index])

# Worst Selling Product
worst_index = np.argmin(sales)

print("\nWorst Selling Product:")
print("Product ID:", product_id[worst_index])
print("Sales:", sales[worst_index])

# Top 5 Products Based on Revenue
top5_index = np.argsort(revenue)[-5:][::-1]

print("\nTOP 5 PRODUCTS BY REVENUE")

for index in top5_index:
    print(
        "Product ID:", product_id[index],
        "| Revenue:", revenue[index]
    )