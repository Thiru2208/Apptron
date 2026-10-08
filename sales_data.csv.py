import pandas as pd
import random
from datetime import datetime, timedelta

products = [
    "Laptop", "Mouse", "Keyboard", "Monitor",
    "Headphones", "Printer", "Tablet", "Phone"
]

categories = ["Electronics", "Accessories", "Office"]

cities = [
    "Colombo", "Jaffna", "Kandy",
    "Galle", "Kurunegala"
]

sales_data = []

start_date = datetime(2026, 1, 1)

for i in range(1, 101):

    product = random.choice(products)
    category = random.choice(categories)
    quantity = random.randint(1, 20)
    price = random.randint(1000, 20000)

    revenue = quantity * price

    city = random.choice(cities)

    random_days = random.randint(0, 364)
    date = start_date + timedelta(days=random_days)

    sales_data.append({
        "Sale_ID": i,
        "Product": product,
        "Category": category,
        "Quantity": quantity,
        "Price": price,
        "Revenue": revenue,
        "City": city,
        "Date": date.strftime("%Y-%m-%d")
    })


df = pd.DataFrame(sales_data)


# Save CSV in Apptron folder
file_path = r"C:\Users\LENOVO\Apptron\sales_data.csv"

df.to_csv(file_path, index=False)


print("Sales data generated successfully!")
print("File saved as:", file_path)
print("Total records generated:", len(df))

print("\nFirst 5 records:")
print(df.head())