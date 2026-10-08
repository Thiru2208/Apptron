import matplotlib.pyplot as plt
import numpy as np

 # BUSINESS KPI DASHBOARD
 
# 1. Create 12 months of company data

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

revenue = [
    120000, 135000, 128000, 145000,
    160000, 175000, 168000, 185000,
    195000, 210000, 225000, 240000
]

profit = [
    25000, 28000, 26000, 32000,
    35000, 40000, 38000, 45000,
    48000, 52000, 58000, 65000
]

orders = [
    320, 350, 340, 390,
    420, 450, 440, 480,
    510, 540, 580, 620
]

customers = [
    200, 230, 250, 280,
    310, 340, 365, 390,
    420, 455, 490, 530
]

expenses = [
    95000, 107000, 102000, 113000,
    125000, 135000, 130000, 140000,
    147000, 158000, 167000, 175000
]


 # 2. Find the Best-Performing Month
 
best_month_index = np.argmax(profit)

best_month = months[best_month_index]
best_profit = profit[best_month_index]

print("BUSINESS KPI DASHBOARD")
print("-" * 40)

print("Best Performing Month:", best_month)
print("Highest Profit:", best_profit)


 # 3. Create Subplots
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))


 # 4. Revenue Trend
 
axes[0, 0].plot(
    months,
    revenue,
    marker="o",
    linewidth=2,
    label="Revenue"
)

axes[0, 0].set_title("Monthly Revenue Trend")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Revenue")
axes[0, 0].legend()
axes[0, 0].grid(True)


# Highlight best-performing month

axes[0, 0].scatter(
    months[best_month_index],
    revenue[best_month_index],
    s=150,
    label="Best Month"
)

axes[0, 0].legend()


 # 5. Profit Trend
 
axes[0, 1].plot(
    months,
    profit,
    marker="o",
    linewidth=2,
    label="Profit"
)

axes[0, 1].set_title("Monthly Profit Trend")
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Profit")
axes[0, 1].legend()
axes[0, 1].grid(True)


# Highlight highest profit

axes[0, 1].scatter(
    months[best_month_index],
    profit[best_month_index],
    s=150,
    label="Highest Profit"
)

axes[0, 1].legend()


 # 6. Monthly Orders
 
axes[1, 0].bar(
    months,
    orders,
    label="Orders"
)

axes[1, 0].set_title("Monthly Orders")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Number of Orders")
axes[1, 0].legend()


 # 7. Customer Growth
 
axes[1, 1].plot(
    months,
    customers,
    marker="o",
    linewidth=2,
    label="Customers"
)

axes[1, 1].set_title("Customer Growth")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Number of Customers")
axes[1, 1].legend()
axes[1, 1].grid(True)


 # 8. Main Dashboard Title
 
fig.suptitle(
    "Business KPI Dashboard",
    fontsize=18,
    fontweight="bold"
)


 # 9. Adjust spacing
 
plt.tight_layout(rect=[0, 0, 1, 0.95])


 # 10. Display Dashboard
 
plt.show()