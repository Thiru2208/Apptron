
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Day 66 - Advanced Cluster Visualization

np.random.seed(42)

# Load customer dataset
if os.path.exists("customer_clusters.csv"):
    df = pd.read_csv("customer_clusters.csv")
    print("Dataset loaded: customer_clusters.csv")

elif os.path.exists("customers.csv"):
    df = pd.read_csv("customers.csv")
    print("Dataset loaded: customers.csv")

else:
    df = pd.DataFrame({
        "CustomerID": range(1, 201),
        "Age": np.random.randint(18, 70, 200),
        "Annual Income (k$)": np.random.randint(15, 140, 200),
        "Spending Score (1-100)": np.random.randint(1, 101, 200),
        "Purchase Frequency": np.random.randint(1, 30, 200),
        "Average Order Value": np.random.randint(20, 500, 200),
        "Website Visits": np.random.randint(1, 60, 200)
    })

    df.to_csv("day66_customers.csv", index=False)
    print("Sample dataset created")

# Correct common column names
df.rename(columns={
    "Customer ID": "CustomerID",
    "Customer_ID": "CustomerID",
    "Annual_Income": "Annual Income (k$)",
    "Annual Income": "Annual Income (k$)",
    "Spending_Score": "Spending Score (1-100)",
    "Spending Score": "Spending Score (1-100)",
    "Purchase_Frequency": "Purchase Frequency",
    "Average_Order_Value": "Average Order Value",
    "Website_Visits": "Website Visits",
    "cluster": "Cluster"
}, inplace=True)

# Create customer ID if missing
if "CustomerID" not in df.columns:
    df["CustomerID"] = range(1, len(df) + 1)

# Check required columns
required_columns = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError("Missing column: " + col)

# Add sample features if the dataset has fewer than 4
sample_columns = []

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

features = [
    col for col in numeric_columns
    if col not in ["CustomerID", "Cluster"]
]

extra_features = {
    "Age": (18, 70),
    "Purchase Frequency": (1, 30),
    "Average Order Value": (20, 500),
    "Website Visits": (1, 60),
    "Product Views": (1, 100)
}

for col, values in extra_features.items():

    if len(features) >= 4:
        break

    if col not in df.columns:
        df[col] = np.random.randint(
            values[0],
            values[1],
            len(df)
        )

        features.append(col)
        sample_columns.append(col)

if sample_columns:
    print("\nSample columns added:")
    print(sample_columns)
    print("Note: These values are randomly generated.")

# Select numerical features
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

features = [
    col for col in numeric_columns
    if col not in ["CustomerID", "Cluster", "PC1", "PC2", "PC3"]
]

# Remove missing values
df = df.dropna(subset=features).copy()

if len(df) < 4:
    raise ValueError("Not enough customer records.")

print("\nFirst 10 Records:")
print(df.head(10))

print("\nSelected Features:")
print(features)

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# K-Means clustering
if "Cluster" in df.columns and df["Cluster"].notna().all():
    print("\nExisting cluster labels used.")

else:
    kmeans = KMeans(
        n_clusters=min(5, len(df)),
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(scaled_data)

    print("\nK-Means clustering completed.")

df["Cluster Name"] = df["Cluster"].astype(str)

print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# Apply PCA
pca = PCA(n_components=2)

pca_result = pca.fit_transform(scaled_data)

df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

# Explained variance
variance = pca.explained_variance_ratio_

pc1 = variance[0] * 100
pc2 = variance[1] * 100
total_variance = pc1 + pc2

print("\nExplained Variance Report")
print("PC1:", round(pc1, 2), "%")
print("PC2:", round(pc2, 2), "%")
print("Total:", round(total_variance, 2), "%")

# Create results folder
folder = "Day66_results"
os.makedirs(folder, exist_ok=True)

# 2D PCA scatter plot
plt.figure(figsize=(9, 6))

for cluster in sorted(df["Cluster Name"].unique()):

    group = df[df["Cluster Name"] == cluster]

    plt.scatter(
        group["PC1"],
        group["PC2"],
        label="Cluster " + cluster,
        alpha=0.7
    )

plt.title("PCA Customer Cluster Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(folder + "/pca_2d.png")
plt.show()

# Original feature scatter plot
plt.figure(figsize=(9, 6))

for cluster in sorted(df["Cluster Name"].unique()):

    group = df[df["Cluster Name"] == cluster]

    plt.scatter(
        group["Annual Income (k$)"],
        group["Spending Score (1-100)"],
        label="Cluster " + cluster,
        alpha=0.7
    )

plt.title("Original Customer Cluster Visualization")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(folder + "/original_scatter.png")
plt.show()

# Interactive Plotly chart
hover_columns = ["CustomerID"]

if "Age" in df.columns:
    hover_columns.append("Age")

hover_columns.extend([
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Cluster"
])

fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Cluster Name",
    hover_data=hover_columns,
    title="Interactive Customer Cluster Explorer",
    labels={"Cluster Name": "Cluster"}
)

fig.write_html(folder + "/interactive_clusters.html")
fig.show()

# Compare original features and PCA
fig_compare, axes = plt.subplots(1, 2, figsize=(13, 5))

for cluster in sorted(df["Cluster Name"].unique()):

    group = df[df["Cluster Name"] == cluster]

    axes[0].scatter(
        group["Annual Income (k$)"],
        group["Spending Score (1-100)"],
        label="Cluster " + cluster,
        alpha=0.7
    )

    axes[1].scatter(
        group["PC1"],
        group["PC2"],
        label="Cluster " + cluster,
        alpha=0.7
    )

axes[0].set_title("Original Features")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score (1-100)")

axes[1].set_title("PCA Features")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(folder + "/comparison.png")
plt.show()

# Bonus - 3D PCA visualization
pca3 = PCA(n_components=3)

pca3_result = pca3.fit_transform(scaled_data)

df["PC1_3D"] = pca3_result[:, 0]
df["PC2_3D"] = pca3_result[:, 1]
df["PC3"] = pca3_result[:, 2]

fig3d = px.scatter_3d(
    df,
    x="PC1_3D",
    y="PC2_3D",
    z="PC3",
    color="Cluster Name",
    hover_data=hover_columns,
    title="3D PCA Customer Clusters",
    labels={"Cluster Name": "Cluster"}
)

fig3d.write_html(folder + "/pca_3d.html")
fig3d.show()

# Five observations
cluster_counts = df["Cluster Name"].value_counts()

largest_cluster = cluster_counts.idxmax()
smallest_cluster = cluster_counts.idxmin()

income_average = df.groupby(
    "Cluster Name"
)["Annual Income (k$)"].mean()

spending_average = df.groupby(
    "Cluster Name"
)["Spending Score (1-100)"].mean()

observations = [
    "1. The dataset contains " + str(len(df)) +
    " customers in " + str(df["Cluster Name"].nunique()) +
    " clusters.",

    "2. Cluster " + largest_cluster +
    " has the highest number of customers (" +
    str(cluster_counts.max()) + ").",

    "3. Cluster " + smallest_cluster +
    " has the lowest number of customers (" +
    str(cluster_counts.min()) + ").",

    "4. PC1 explains " + str(round(pc1, 2)) +
    "% and PC2 explains " + str(round(pc2, 2)) +
    "% of total variance.",

    "5. Cluster " + spending_average.idxmax() +
    " has the highest average spending score (" +
    str(round(spending_average.max(), 2)) + ")."
]

print("\nFive Observations")

for observation in observations:
    print(observation)

# Comparison
comparison = (
    "The original scatter plot displays Annual Income and "
    "Spending Score directly. The PCA scatter plot combines "
    "all selected numerical features into two components. "
    "PCA helps visualize patterns across multiple features."
)

# PCA limitation
limitation = (
    "PCA reduces the number of dimensions, so some original "
    "information may be lost. Clusters may overlap in the "
    "2D PCA plot even when they are separated in the "
    "original feature space."
)

print("\nComparison:")
print(comparison)

print("\nPCA Limitation:")
print(limitation)

# Save analysis report
report = [
    "Day 66 - Advanced Cluster Explorer",
    "",
    "Selected Features: " + ", ".join(features),
    "PC1 Explained Variance: " + str(round(pc1, 2)) + "%",
    "PC2 Explained Variance: " + str(round(pc2, 2)) + "%",
    "Total Explained Variance: " + str(round(total_variance, 2)) + "%",
    "",
    "Sample Features Added: " +
    (", ".join(sample_columns) if sample_columns else "None"),
    "",
    "Five Observations:"
]

report.extend(observations)

report.extend([
    "",
    "Comparison:",
    comparison,
    "",
    "PCA Limitation:",
    limitation
])

with open(
    folder + "/analysis_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(report))

# Save final dataset
df.to_csv(
    folder + "/customer_pca_results.csv",
    index=False
)

print("\nAll results saved successfully!")
print("Output Folder:", os.path.abspath(folder))
