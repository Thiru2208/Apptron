import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Set output folder
output_folder = Path(__file__).resolve().parent / "Day59_results"
output_folder.mkdir(exist_ok=True)

# Load customer dataset
dataset_path = Path(__file__).resolve().parent / "Mall_Customers.csv"

if dataset_path.exists():
    df = pd.read_csv(dataset_path)
    print("Existing customer dataset loaded successfully.")
else:
    # Create sample customer dataset if CSV is unavailable
    np.random.seed(42)

    df = pd.DataFrame({
        "CustomerID": range(1, 201),
        "Annual Income (k$)": np.random.randint(15, 140, 200),
        "Spending Score (1-100)": np.random.randint(1, 101, 200)
    })

    df.to_csv(dataset_path, index=False)
    print("Sample customer dataset created successfully.")

print("\nFirst 10 Records:")
print(df.head(10))

# Select features
features = ["Annual Income (k$)", "Spending Score (1-100)"]

X = df[features].copy()

# Check feature scales
print("\nFeature Statistics:")
print(X.describe())

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFirst 10 Scaled Records:")
print(pd.DataFrame(X_scaled, columns=features).head(10))

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train model and predict clusters
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display first 10 customers
print("\nFirst 10 Customers with Cluster Labels:")
print(df.head(10))

# Display cluster counts
print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# Display scaled centroids
print("\nCluster Centroids (Scaled Values):")
print(kmeans.cluster_centers_)

# Convert centroids to original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

print("\nCluster Centroids (Original Values):")
for i, centroid in enumerate(centroids):
    print(
        f"Cluster {i}: "
        f"Income = {centroid[0]:.2f}, "
        f"Spending Score = {centroid[1]:.2f}"
    )

# Plot clusters in scaled feature space
plt.figure(figsize=(10, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=60,
    alpha=0.7
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Segmentation Using K-Means (K=3)")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid(True)

plt.savefig(output_folder / "KMeans_Clusters.png", dpi=300)
plt.show()

# Save dataset with cluster labels
df.to_csv(
    output_folder / "Customers_With_Clusters.csv",
    index=False
)

# Display model inertia
print("\nModel Inertia:", kmeans.inertia_)

# Explain clusters based on centroids
print("\nCluster Interpretation:")

for i, centroid in enumerate(centroids):
    income = centroid[0]
    spending = centroid[1]

    print(f"\nCluster {i}:")

    if income >= X.iloc[:, 0].median():
        print("Higher annual income")
    else:
        print("Lower annual income")

    if spending >= X.iloc[:, 1].median():
        print("Higher spending score")
    else:
        print("Lower spending score")

# Bonus: Compare K=2, K=3 and K=4
print("\nComparing Different K Values:")

for k in [2, 3, 4]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    print(f"K={k}, Inertia={model.inertia_:.2f}")

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7
    )

    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        c="red",
        marker="X",
        s=250,
        label="Centroids"
    )

    plt.title(f"K-Means Clustering (K={k})")
    plt.xlabel("Annual Income (Scaled)")
    plt.ylabel("Spending Score (Scaled)")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_folder / f"KMeans_K{k}.png",
        dpi=300
    )

    plt.show()

print("\nAll results saved successfully!")
print("Output Folder:", output_folder)