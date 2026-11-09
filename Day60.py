
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load customer dataset
folder = Path(__file__).resolve().parent

file_path = folder / "Mall_Customers.csv"

if not file_path.exists():
    file_path = folder / "Day59_customers.csv"

if not file_path.exists():
    print("Customer dataset not found.")
    print("Place your Day 59 CSV file in the Apptron folder.")
    raise SystemExit

df = pd.read_csv(file_path)

print("First 10 Records:")
print(df.head(10))

# Select Annual Income and Spending Score
income_col = next(
    (c for c in df.columns if "income" in c.lower()),
    None
)

spending_col = next(
    (c for c in df.columns if "spending" in c.lower()),
    None
)

if income_col is None or spending_col is None:
    print("Required columns not found.")
    raise SystemExit

data = df[[income_col, spending_col]].copy()
data = data.apply(pd.to_numeric, errors="coerce").dropna()

if len(data) < 11:
    print("At least 11 valid customer records are required.")
    raise SystemExit

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Elbow analysis function
def elbow_analysis(X):
    inertia = []
    silhouette_scores = []

    for k in range(1, 11):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)
        inertia.append(model.inertia_)

        if 1 < len(set(labels)) < len(X):
            score = silhouette_score(X, labels)
        else:
            score = np.nan

        silhouette_scores.append(score)

    return inertia, silhouette_scores


# Run elbow analysis
inertia, silhouette_scores = elbow_analysis(scaled_data)

# Display inertia and silhouette table
results = pd.DataFrame({
    "K": range(1, 11),
    "Inertia": inertia,
    "Silhouette Score": silhouette_scores
})

print("\nElbow Analysis Results:")
print(results.round(4).to_string(index=False))

# Automatically identify candidate elbow
points = np.column_stack((np.arange(1, 11), inertia))

normalized = (points - points.min(axis=0)) / (
    points.max(axis=0) - points.min(axis=0)
)

start = normalized[0]
end = normalized[-1]

# Corrected distance calculation for NumPy 2.x
line_vector = end - start
point_vectors = normalized - start

distances = np.abs(
    line_vector[0] * point_vectors[:, 1]
    - line_vector[1] * point_vectors[:, 0]
) / np.linalg.norm(line_vector)

best_k = int(np.argmax(distances) + 1)

print("\nSuggested K from Elbow Method:", best_k)

# Plot Elbow Method
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o",
    linestyle="-"
)

plt.axvline(
    best_k,
    color="red",
    linestyle="--",
    label=f"Selected K = {best_k}"
)

plt.title("Elbow Method for K-Means Clustering")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Train final K-Means model
final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = final_model.fit_predict(scaled_data)

data["Cluster"] = clusters

print("\nCluster Counts:")
print(data["Cluster"].value_counts().sort_index())

# Plot customer clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    data[income_col],
    data[spending_col],
    c=clusters,
    cmap="viridis",
    s=50
)

# Convert centroids to original feature scale
centers = scaler.inverse_transform(
    final_model.cluster_centers_
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.title(f"Customer Segmentation (K = {best_k})")
plt.xlabel(income_col)
plt.ylabel(spending_col)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Bonus: Silhouette Score Comparison
plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores[1:],
    marker="o"
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 11))
plt.grid(True)

plt.tight_layout()
plt.show()

# Find best K using silhouette score
best_silhouette_k = int(
    np.nanargmax(silhouette_scores) + 1
)

print("\nBest K from Silhouette Score:", best_silhouette_k)

print("\nExplanation:")
print(
    f"K = {best_k} was selected as a candidate using "
    "the Elbow Method because the reduction in "
    "inertia begins to slow around this point."
)

print(
    "The elbow is not always definitive because "
    "some datasets do not have a clear bend."
)

print(
    f"The silhouette method suggests K = "
    f"{best_silhouette_k}."
)

if best_k == best_silhouette_k:
    print("Both methods suggest the same K.")
else:
    print("The two methods suggest different K values.")

print("\nDay 60 Task Completed Successfully!")
