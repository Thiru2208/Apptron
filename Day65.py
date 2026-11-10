
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# Create output folder
os.makedirs("Day65_results", exist_ok=True)

# Load dataset
if os.path.exists("customer_clusters.csv"):
    df = pd.read_csv("customer_clusters.csv")
else:
    print("Dataset not found. Creating sample data...")

    np.random.seed(42)

    df = pd.DataFrame({
        "CustomerID": range(1, 201),
        "Age": np.random.randint(18, 70, 200),
        "Annual Income (k$)": np.random.randint(15, 140, 200),
        "Spending Score (1-100)": np.random.randint(1, 101, 200)
    })

# Column names
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

# Create clusters if not already available
if "Cluster" not in df.columns:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(scaled_data)

df.to_csv("Day65_results/customer_clusters.csv", index=False)

print("\nFirst 10 Records:")
print(df.head(10))

# Calculate cluster size
cluster_size = df["Cluster"].value_counts().sort_index()

# Calculate percentages
cluster_percentage = cluster_size / len(df) * 100

size_table = pd.DataFrame({
    "Customers": cluster_size,
    "Percentage": cluster_percentage.round(2)
})

print("\nCluster Size and Percentage:")
print(size_table)

size_table.to_csv("Day65_results/cluster_size.csv")

# Calculate average values
cluster_profile = df.groupby("Cluster")[features].mean().round(2)

print("\nCluster Average Profile:")
print(cluster_profile)

cluster_profile.to_csv("Day65_results/cluster_profile.csv")

# Cluster size bar chart
plt.figure(figsize=(8, 5))

plt.bar(cluster_size.index.astype(str), cluster_size.values, color="skyblue")

plt.title("Customer Cluster Size")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("Day65_results/cluster_size.png")
plt.show()

# Cluster profile heatmap
heatmap_scaler = StandardScaler()
heatmap_data = heatmap_scaler.fit_transform(cluster_profile)

plt.figure(figsize=(9, 5))

plt.imshow(heatmap_data, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Standardized Value")

plt.xticks(range(len(features)), features, rotation=20)
plt.yticks(range(len(cluster_profile)), cluster_profile.index)

plt.title("Cluster Profile Heatmap")

for i in range(len(cluster_profile)):
    for j in range(len(features)):
        plt.text(j, i, round(heatmap_data[i, j], 2),
                 ha="center", va="center")

plt.tight_layout()
plt.savefig("Day65_results/cluster_heatmap.png")
plt.show()

# Grouped bar chart
income = cluster_profile["Annual Income (k$)"]
spending = cluster_profile["Spending Score (1-100)"]

x = np.arange(len(cluster_profile))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(x - width / 2, income, width, label="Annual Income")
plt.bar(x + width / 2, spending, width, label="Spending Score")

plt.xticks(x, cluster_profile.index)
plt.xlabel("Cluster")
plt.ylabel("Average Value")
plt.title("Average Income and Spending Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("Day65_results/grouped_bar_chart.png")
plt.show()

# Radar chart
radar_scaler = MinMaxScaler()
radar_data = radar_scaler.fit_transform(cluster_profile)

labels = ["Age", "Income", "Spending"]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8, 7))
ax = plt.subplot(111, polar=True)

for i in range(len(cluster_profile)):
    values = radar_data[i].tolist()
    values += values[:1]

    ax.plot(angles, values, marker="o",
            label="Cluster " + str(cluster_profile.index[i]))
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)

plt.title("Customer Cluster Radar Chart")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig("Day65_results/radar_chart.png")
plt.show()

# Largest and smallest clusters
largest = cluster_size.idxmax()
smallest = cluster_size.idxmin()

print("\nLargest Cluster:", largest)
print("Customers:", cluster_size[largest])

print("\nSmallest Cluster:", smallest)
print("Customers:", cluster_size[smallest])

# Give meaningful cluster names
avg_income = df["Annual Income (k$)"].mean()
avg_spending = df["Spending Score (1-100)"].mean()

cluster_names = {}

for cluster in cluster_profile.index:
    income_value = cluster_profile.loc[cluster, "Annual Income (k$)"]
    spending_value = cluster_profile.loc[cluster, "Spending Score (1-100)"]

    if income_value >= avg_income and spending_value >= avg_spending:
        name = "High Income High Spending"

    elif income_value >= avg_income and spending_value < avg_spending:
        name = "High Income Low Spending"

    elif income_value < avg_income and spending_value >= avg_spending:
        name = "Low Income High Spending"

    else:
        name = "Low Income Low Spending"

    cluster_names[cluster] = name

df["Cluster Name"] = df["Cluster"].map(cluster_names)

print("\nCluster Names:")

for cluster, name in cluster_names.items():
    print("Cluster", cluster, ":", name)

df.to_csv("Day65_results/customer_cluster_names.csv", index=False)

# Business insights
insights = []

insights.append(
    f"1. Cluster {largest} has the highest number of customers ({cluster_size[largest]})."
)

insights.append(
    f"2. Cluster {smallest} has the lowest number of customers ({cluster_size[smallest]})."
)

high_income = cluster_profile["Annual Income (k$)"].idxmax()
insights.append(
    f"3. Cluster {high_income} has the highest average annual income."
)

high_spending = cluster_profile["Spending Score (1-100)"].idxmax()
insights.append(
    f"4. Cluster {high_spending} has the highest average spending score."
)

low_spending = cluster_profile["Spending Score (1-100)"].idxmin()
insights.append(
    f"5. Cluster {low_spending} has the lowest spending score and may need special offers."
)

print("\nBusiness Insights:")

for insight in insights:
    print(insight)

with open("Day65_results/business_insights.txt", "w") as file:
    for insight in insights:
        file.write(insight + "\n")

# Bonus: Median comparison
cluster_median = df.groupby("Cluster")[features].median().round(2)

print("\nCluster Median Profile:")
print(cluster_median)

cluster_median.to_csv("Day65_results/cluster_median.csv")

print("\nAverage and Median Comparison:")

for column in features:
    print("\n", column)

    comparison = pd.DataFrame({
        "Mean": cluster_profile[column],
        "Median": cluster_median[column]
    })

    comparison["Difference"] = (
        comparison["Mean"] - comparison["Median"]
    ).round(2)

    print(comparison)

print("\nAll Day 65 tasks completed successfully!")
print("Results saved in Day65_results folder.")
