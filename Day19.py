import pandas as pd
import numpy as np

 
# CUSTOMER SEGMENTATION SYSTEM
 

# Set random seed
np.random.seed(10)

# Number of customers
n = 1000

# Create Dataset
customer_data = pd.DataFrame({

    "Customer_ID": np.arange(1, n + 1),

    "Age": np.random.randint(18, 71, n),

    "Gender": np.random.choice(
        ["Male", "Female"],
        n
    ),

    "City": np.random.choice(
        ["Colombo", "Kandy", "Galle", "Jaffna", "Kurunegala"],
        n
    ),

    "Income": np.random.randint(
        30000,
        200001,
        n
    ),

    "Purchase_Amount": np.random.randint(
        0,
        100001,
        n
    ),

    "Product_Category": np.random.choice(
        [
            "Electronics",
            "Clothing",
            "Groceries",
            "Furniture",
            "Accessories"
        ],
        n
    )
})


 
# DISPLAY DATASET
 

print("\nCUSTOMER DATASET")
print("-" * 70)

print(customer_data)

print("\nTotal number of customers:", len(customer_data))


 
# 1. CUSTOMERS ABOVE AGE 40
 

customers_above_40 = customer_data[
    customer_data["Age"] > 40
]

print("\nCUSTOMERS ABOVE AGE 40")
print("-" * 70)

print(customers_above_40)

print(
    "Total customers above age 40:",
    len(customers_above_40)
)


 
# 2. CUSTOMERS WITH INCOME > 100,000
 

high_income_customers = customer_data[
    customer_data["Income"] > 100000
]

print("\nCUSTOMERS WITH INCOME ABOVE 100,000")
print("-" * 70)

print(high_income_customers)

print(
    "Total high-income customers:",
    len(high_income_customers)
)


 
# 3. CUSTOMERS SPENDING > 50,000
 

high_spending_customers = customer_data[
    customer_data["Purchase_Amount"] > 50000
]

print("\nCUSTOMERS SPENDING ABOVE 50,000")
print("-" * 70)

print(high_spending_customers)

print(
    "Total high-spending customers:",
    len(high_spending_customers)
)


 
# 4. COLOMBO CUSTOMERS
 

colombo_customers = customer_data[
    customer_data["City"] == "Colombo"
]

print("\nCOLOMBO CUSTOMERS")
print("-" * 70)

print(colombo_customers)

print(
    "Total Colombo customers:",
    len(colombo_customers)
)


 
# 5. HIGH-INCOME + HIGH-SPENDING CUSTOMERS
 

premium_customers = customer_data[
    (customer_data["Income"] > 100000)
    &
    (customer_data["Purchase_Amount"] > 50000)
]

print("\nHIGH-INCOME + HIGH-SPENDING CUSTOMERS")
print("-" * 70)

print(premium_customers)

print(
    "Total high-income + high-spending customers:",
    len(premium_customers)
)


 
# 6. CUSTOMERS WHO HAVE NEVER PURCHASED
 

never_purchased = customer_data[
    customer_data["Purchase_Amount"] == 0
]

print("\nCUSTOMERS WHO HAVE NEVER PURCHASED")
print("-" * 70)

print(never_purchased)

print(
    "Total customers who never purchased:",
    len(never_purchased)
)


 
# BONUS
# CREATE CUSTOMER CATEGORIES
 

def customer_category(amount):

    if amount > 50000:
        return "Premium"

    elif amount >= 20000:
        return "Regular"

    else:
        return "Low Value"


customer_data["Customer_Category"] = (
    customer_data["Purchase_Amount"]
    .apply(customer_category)
)


 
# DISPLAY CUSTOMER CATEGORIES
 

print("\nCUSTOMER DATASET WITH CATEGORIES")
print("-" * 70)

print(customer_data)


 
# CATEGORY COUNTS
 

print("\nCUSTOMER CATEGORY COUNTS")
print("-" * 70)

print(
    customer_data[
        "Customer_Category"
    ].value_counts()
)


 
# DISPLAY FIRST 10 RECORDS
 

print("\nFIRST 10 CUSTOMER RECORDS")
print("-" * 70)

print(customer_data.head(10))