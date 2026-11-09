
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load customer dataset
try:
    df = pd.read_csv("customers.csv")
except FileNotFoundError:
    np.random.seed(42)

    df = pd.DataFrame({
        "CustomerID": range(1, 201),
        "Age": np.random.randint(18, 65, 200),
        "Annual Income (k$)": np.random.randint(15, 140, 200),
        "Spending Score (1-100)": np.random.randint(1, 101, 200)
    })

    df.to_csv("customers.csv", index=False)

print("First 10 Records:")
print(df.head(10))

# Select columns for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
k = 5

model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

print("\nCustomer Cluster Details:")
print(df.head(10))

# Customer count and percentage
count = df["Cluster"].value_counts().sort_index()
percentage = (count / len(df)) * 100

print("\nCluster Count:")
print(count)

print("\nCluster Percentage:")
print(percentage.round(2))

# Average values for each cluster
columns = ["Annual Income (k$)", "Spending Score (1-100)"]

if "Age" in df.columns:
    columns.insert(0, "Age")

profile = df.groupby("Cluster")[columns].mean()

profile["Count"] = count
profile["Percentage"] = percentage

print("\nCluster Profile:")
print(profile.round(2))

# Give names to clusters
income_avg = df["Annual Income (k$)"].mean()
spending_avg = df["Spending Score (1-100)"].mean()

names = {}

for i in range(k):
    income = profile.loc[i, "Annual Income (k$)"]
    spending = profile.loc[i, "Spending Score (1-100)"]

    if income >= income_avg and spending >= spending_avg:
        names[i] = "High Income High Spending"

    elif income >= income_avg and spending < spending_avg:
        names[i] = "High Income Low Spending"

    elif income < income_avg and spending >= spending_avg:
        names[i] = "Low Income High Spending"

    else:
        names[i] = "Low Income Low Spending"

df["Segment"] = df["Cluster"].map(names)
profile["Segment"] = profile.index.map(names)

print("\nCustomer Segments:")
print(profile.round(2))

# Business recommendations
print("\nBusiness Recommendations:")

for i in range(k):
    print("\nCluster", i, "-", names[i])

    if names[i] == "High Income High Spending":
        print("1. Offer premium products")
        print("2. Give loyalty rewards")

    elif names[i] == "High Income Low Spending":
        print("1. Give special offers")
        print("2. Recommend suitable products")

    elif names[i] == "Low Income High Spending":
        print("1. Give discounts")
        print("2. Offer affordable product bundles")

    else:
        print("1. Promote budget products")
        print("2. Use low-cost marketing")

# Scatter plot
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.colorbar(label="Cluster")
plt.show()

# Average income chart
plt.figure(figsize=(8, 5))
profile["Annual Income (k$)"].plot(kind="bar", color="skyblue")
plt.xlabel("Cluster")
plt.ylabel("Average Income")
plt.title("Average Income by Cluster")
plt.xticks(rotation=0)
plt.show()

# Average spending chart
plt.figure(figsize=(8, 5))
profile["Spending Score (1-100)"].plot(kind="bar", color="orange")
plt.xlabel("Cluster")
plt.ylabel("Average Spending Score")
plt.title("Average Spending by Cluster")
plt.xticks(rotation=0)
plt.show()

# Customer count chart
plt.figure(figsize=(8, 5))
count.plot(kind="bar", color="lightgreen")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("Customers in Each Cluster")
plt.xticks(rotation=0)
plt.show()

# Findings
print("\nFindings:")

largest = count.idxmax()
high_income = profile["Annual Income (k$)"].idxmax()
high_spending = profile["Spending Score (1-100)"].idxmax()

print("1. Cluster", largest, "has the most customers.")
print("2. Cluster", high_income, "has the highest average income.")
print("3. Cluster", high_spending, "has the highest average spending.")

# Limitation
print("\nLimitation:")
print("K-Means only groups customers based on selected features.")
print("Other customer behaviours are not considered.")

# Save results
df.to_csv("Day61_customers.csv", index=False)
profile.to_csv("Day61_profile.csv")

print("\nFiles saved successfully!")
