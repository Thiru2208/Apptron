import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# CUSTOMER BEHAVIOR ANALYZER
# ---------------------------------------------------

# Same random values every time
np.random.seed(42)

# Number of customers
n = 1000


# ---------------------------------------------------
# 1. CREATE RANDOM CUSTOMER DATA
# ---------------------------------------------------

age = np.random.randint(18, 70, n)

income = np.random.randint(30000, 200001, n)

purchase_amount = np.random.randint(500, 10001, n)

website_visits = np.random.randint(1, 31, n)

time_spent = np.random.randint(5, 121, n)

customer_rating = np.round(
    np.random.uniform(1, 5, n),
    1
)


# ---------------------------------------------------
# 2. CREATE DATAFRAME
# ---------------------------------------------------

customers = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Purchase_Amount": purchase_amount,
    "Website_Visits": website_visits,
    "Time_Spent": time_spent,
    "Customer_Rating": customer_rating
})


# ---------------------------------------------------
# 3. DISPLAY DATA
# ---------------------------------------------------

print("=" * 60)
print("CUSTOMER BEHAVIOR ANALYZER")
print("=" * 60)

print("\nFIRST 10 CUSTOMER RECORDS")
print("-" * 60)
print(customers.head(10))


print("\nDATASET SHAPE")
print("-" * 60)
print(customers.shape)


print("\nSTATISTICAL SUMMARY")
print("-" * 60)
print(customers.describe())


# ---------------------------------------------------
# 4. SAVE DATASET
# ---------------------------------------------------

customers.to_csv(
    r"C:\Users\LENOVO\Apptron\customer_behavior_data.csv",
    index=False
)

print("\nDataset saved successfully as customer_behavior_data.csv")


# ---------------------------------------------------
# 5. SCATTER PLOT
# Income vs Purchase Amount
# ---------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    customers["Income"],
    customers["Purchase_Amount"],
    alpha=0.6
)

plt.title("Income vs Purchase Amount")
plt.xlabel("Income")
plt.ylabel("Purchase Amount")

plt.grid(True)
plt.show()


# ---------------------------------------------------
# 6. HISTOGRAM
# Customer Age
# ---------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    customers["Age"],
    bins=10,
    edgecolor="black"
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.grid(axis="y")
plt.show()


# ---------------------------------------------------
# 7. HISTOGRAM
# Purchase Amount
# ---------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    customers["Purchase_Amount"],
    bins=20,
    edgecolor="black"
)

plt.title("Purchase Amount Distribution")
plt.xlabel("Purchase Amount")
plt.ylabel("Frequency")

plt.grid(axis="y")
plt.show()


# ---------------------------------------------------
# 8. SCATTER PLOT
# Website Visits vs Purchase Amount
# ---------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    customers["Website_Visits"],
    customers["Purchase_Amount"],
    alpha=0.6
)

plt.title("Website Visits vs Purchase Amount")
plt.xlabel("Website Visits")
plt.ylabel("Purchase Amount")

plt.grid(True)
plt.show()


# ---------------------------------------------------
# 9. CORRELATION
# ---------------------------------------------------

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

income_correlation = customers["Income"].corr(
    customers["Purchase_Amount"]
)

website_correlation = customers["Website_Visits"].corr(
    customers["Purchase_Amount"]
)

print(
    "\nIncome vs Purchase Amount:",
    round(income_correlation, 3)
)

print(
    "Website Visits vs Purchase Amount:",
    round(website_correlation, 3)
)


# ---------------------------------------------------
# 10. TOP 10 HIGHEST SPENDING CUSTOMERS
# ---------------------------------------------------

top_spenders = customers.sort_values(
    by="Purchase_Amount",
    ascending=False
).head(10)

print("\n" + "=" * 60)
print("TOP 10 HIGHEST SPENDING CUSTOMERS")
print("=" * 60)

print(top_spenders)


# ---------------------------------------------------
# 11. CREATE AGE GROUPS
# ---------------------------------------------------

customers["Age_Group"] = pd.cut(
    customers["Age"],
    bins=[17, 25, 35, 45, 55, 70],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-70"
    ]
)


# ---------------------------------------------------
# 12. HIGHEST SPENDING AGE GROUP
# ---------------------------------------------------

age_group_spending = customers.groupby(
    "Age_Group",
    observed=True
)["Purchase_Amount"].mean()

print("\n" + "=" * 60)
print("AVERAGE PURCHASE AMOUNT BY AGE GROUP")
print("=" * 60)

print(age_group_spending)

highest_spending_group = age_group_spending.idxmax()

print(
    "\nHighest Spending Age Group:",
    highest_spending_group
)


# ---------------------------------------------------
# 13. OUTLIER DETECTION USING IQR
# ---------------------------------------------------

Q1 = customers["Purchase_Amount"].quantile(0.25)

Q3 = customers["Purchase_Amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - (1.5 * IQR)

upper_limit = Q3 + (1.5 * IQR)


outliers = customers[
    (customers["Purchase_Amount"] < lower_limit)
    |
    (customers["Purchase_Amount"] > upper_limit)
]


print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

print("Q1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print("Lower Limit:", lower_limit)

print("Upper Limit:", upper_limit)

print(
    "\nNumber of Outliers:",
    len(outliers)
)


if len(outliers) > 0:

    print("\nOutlier Records:")
    print(outliers)

else:

    print("\nNo outliers found.")


# ---------------------------------------------------
# 14. DISTRIBUTION ANALYSIS
# ---------------------------------------------------

age_skewness = customers["Age"].skew()

purchase_skewness = customers["Purchase_Amount"].skew()


print("\n" + "=" * 60)
print("DISTRIBUTION ANALYSIS")
print("=" * 60)


print(
    "\nAge Skewness:",
    round(age_skewness, 3)
)

print(
    "Purchase Amount Skewness:",
    round(purchase_skewness, 3)
)


def check_distribution(skewness):

    if -0.5 <= skewness <= 0.5:

        return "Approximately Symmetric Distribution"

    elif skewness > 0.5:

        return "Right Skewed Distribution"

    else:

        return "Left Skewed Distribution"


print(
    "\nAge Distribution:",
    check_distribution(age_skewness)
)

print(
    "Purchase Amount Distribution:",
    check_distribution(purchase_skewness)
)


# ---------------------------------------------------
# 15. FINAL ANALYSIS
# ---------------------------------------------------

print("\n" + "=" * 60)
print("FINAL ANALYSIS")
print("=" * 60)


print(
    "\n1. Income vs Purchase Amount Correlation:",
    round(income_correlation, 3)
)

if abs(income_correlation) < 0.3:

    print(
        "There is no strong relationship between "
        "income and purchase amount."
    )

else:

    print(
        "There is a relationship between "
        "income and purchase amount."
    )


print(
    "\n2. Website Visits vs Purchase Amount Correlation:",
    round(website_correlation, 3)
)

if abs(website_correlation) < 0.3:

    print(
        "There is no strong relationship between "
        "website visits and purchase amount."
    )

else:

    print(
        "There is a relationship between "
        "website visits and purchase amount."
    )


print(
    "\n3. Highest Spending Age Group:",
    highest_spending_group
)

print(
    "\n4. Number of Purchase Amount Outliers:",
    len(outliers)
)

print(
    "\n5. Age Distribution:",
    check_distribution(age_skewness)
)

print(
    "\n6. Purchase Amount Distribution:",
    check_distribution(purchase_skewness)
)


print("\nCustomer Behavior Analysis Completed Successfully!")