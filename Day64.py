
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load customer dataset
try:
    df = pd.read_csv("customers.csv")
    print("Dataset loaded successfully")
except FileNotFoundError:
    np.random.seed(42)

    df = pd.DataFrame({
        "Age": np.random.randint(18, 65, 200),
        "Annual Income (k$)": np.random.randint(15, 140, 200),
        "Spending Score (1-100)": np.random.randint(1, 101, 200),
        "Purchase Frequency": np.random.randint(1, 30, 200),
        "Average Order Value": np.random.randint(20, 500, 200)
    })

    df.to_csv("customers.csv", index=False)
    print("New customer dataset created")

print("\nFirst 10 Records:")
print(df.head(10))

print("\nDataset Information:")
print(df.info())

# Select numerical features
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

if "Purchase Frequency" in df.columns:
    features.append("Purchase Frequency")

if "Average Order Value" in df.columns:
    features.append("Average Order Value")

# Remove rows with missing values
df = df.dropna(subset=features).copy()

# Create clusters if not available
if "Cluster" not in df.columns:

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    print("\nK-Means clustering completed")

else:
    print("\nExisting cluster labels used")

print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())

# Pair Plot
sns.pairplot(
    df,
    vars=features,
    hue="Cluster",
    palette="viridis",
    diag_kind="hist"
)

plt.suptitle("Customer Cluster Pair Plot", y=1.02)
plt.savefig("Day64_Pairplot.png", bbox_inches="tight")
plt.show()

# Bubble Chart
plt.figure(figsize=(9, 6))

if "Purchase Frequency" in df.columns:
    bubble_size = df["Purchase Frequency"] * 15
else:
    bubble_size = 100

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    s=bubble_size,
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.6,
    edgecolors="black"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Cluster Bubble Chart")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig("Day64_Bubble_Chart.png")
plt.show()

# Histograms for each feature
for feature in features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=feature,
        hue="Cluster",
        bins=15,
        multiple="layer",
        alpha=0.5,
        palette="viridis"
    )

    plt.title(feature + " Distribution by Cluster")
    plt.xlabel(feature)
    plt.ylabel("Customer Count")
    plt.tight_layout()

    filename = "Day64_" + feature.replace(" ", "_").replace("/", "_") + "_Histogram.png"
    plt.savefig(filename)
    plt.show()

# Compare cluster feature averages
cluster_means = df.groupby("Cluster")[features].mean()

print("\nAverage Feature Values by Cluster:")
print(cluster_means.round(2))

# Box plots
for feature in features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Cluster",
        y=feature,
        hue="Cluster",
        palette="viridis",
        legend=False
    )

    plt.title(feature + " Comparison Between Clusters")
    plt.tight_layout()
    plt.show()

# Find features that separate clusters
feature_range = df[features].max() - df[features].min()
mean_difference = cluster_means.max() - cluster_means.min()

separation_score = mean_difference / feature_range.replace(0, np.nan)
separation_score = separation_score.fillna(0).sort_values(ascending=False)

print("\nFeature Separation Scores:")
print(separation_score.round(3))

print("\nBest Feature for Cluster Separation:")
print(separation_score.index[0])

# Correlation Heatmap
plt.figure(figsize=(9, 6))

correlation = df[features].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Customer Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("Day64_Correlation_Heatmap.png")
plt.show()

# Five observations
print("\nFive Observations:")

print("1. Total number of customers:", len(df))

print("2. Number of clusters:", df["Cluster"].nunique())

print("3. Largest cluster:",
      df["Cluster"].value_counts().idxmax())

print("4. Feature with highest separation:",
      separation_score.index[0])

print("5. Feature with lowest separation:",
      separation_score.index[-1])

# Generate analysis report
with open("Day64_Report.txt", "w") as file:

    file.write("DAY 64 - MULTI-FEATURE CLUSTER ANALYSIS\n\n")

    file.write("Total Customers: " + str(len(df)) + "\n")
    file.write("Number of Clusters: " + str(df["Cluster"].nunique()) + "\n\n")

    file.write("Cluster Counts:\n")
    file.write(df["Cluster"].value_counts().sort_index().to_string())

    file.write("\n\nCluster Feature Averages:\n")
    file.write(cluster_means.round(2).to_string())

    file.write("\n\nFeature Separation Scores:\n")
    file.write(separation_score.round(3).to_string())

    file.write("\n\nObservations:\n")
    file.write("1. The dataset contains " + str(len(df)) + " customers.\n")
    file.write("2. Customers are divided into " + str(df["Cluster"].nunique()) + " clusters.\n")
    file.write("3. The largest cluster is " + str(df["Cluster"].value_counts().idxmax()) + ".\n")
    file.write("4. The strongest separating feature is " + separation_score.index[0] + ".\n")
    file.write("5. The weakest separating feature is " + separation_score.index[-1] + ".\n")

    file.write("\nCorrelation Matrix:\n")
    file.write(correlation.round(2).to_string())

df.to_csv("Day64_Clustered_Customers.csv", index=False)

print("\nAll visualizations completed successfully!")
print("Analysis report saved as Day64_Report.txt")
print("Clustered dataset saved successfully")
