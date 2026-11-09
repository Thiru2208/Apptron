
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("First 10 Records:")
print(df.head(10))

# Select required columns
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Find cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Centroids:")
for i in range(5):
    print("Cluster", i, ":", centroids[i])

# Create scatter plot
plt.figure(figsize=(9, 6))

colors = ["red", "blue", "green", "orange", "purple"]

for i in range(5):
    cluster_data = df[df["Cluster"] == i]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        color=colors[i],
        label="Cluster " + str(i),
        alpha=0.7
    )

# Display centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="black",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Clusters Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.savefig("customer_clusters.png")
plt.show()

# Display cluster summary
print("\nCluster Summary:")
summary = df.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean()

summary["Customer Count"] = df.groupby("Cluster").size()

print(summary.round(2))

# Five observations
print("\nFive Observations:")

print("1. Customers are divided into five clusters.")
print("2. Each cluster contains customers with similar characteristics.")
print("3. Some clusters have higher income customers.")
print("4. Some clusters have higher spending scores.")
print("5. Centroids represent the center of each customer group.")

# Bonus Task - Compare different K values
for k in [3, 4, 5]:

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    centers = scaler.inverse_transform(model.cluster_centers_)

    plt.figure(figsize=(8, 5))

    for i in range(k):
        cluster_points = X[labels == i]

        plt.scatter(
            cluster_points["Annual Income (k$)"],
            cluster_points["Spending Score (1-100)"],
            label="Cluster " + str(i)
        )

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        color="black",
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.title("K-Means Clustering - K = " + str(k))
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.grid(True)

    plt.savefig("clusters_k" + str(k) + ".png")
    plt.show()

print("\nAll clustering graphs saved successfully!")
