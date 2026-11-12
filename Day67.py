
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Dashboard", layout="wide")

st.title("Customer Cluster Dashboard")
st.write("Customer Segmentation using K-Means")

# Load customer data
df = pd.read_csv("customers.csv")

df = df.drop_duplicates()

income = "Annual Income (k$)"
spending = "Spending Score (1-100)"

df[income] = pd.to_numeric(df[income], errors="coerce")
df[spending] = pd.to_numeric(df[spending], errors="coerce")
df = df.dropna(subset=[income, spending]).copy()

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())

st.subheader("Customer Dataset")
st.dataframe(df.head(10))

# Select number of clusters
st.sidebar.header("Filters")

k = st.sidebar.selectbox("Number of Clusters", [3, 4])

# K-Means clustering
x = df[[income, spending]]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(x_scaled)

# Select clusters
clusters = sorted(df["Cluster"].unique())

selected = st.sidebar.multiselect(
    "Select Cluster",
    clusters,
    default=clusters
)

filtered = df[df["Cluster"].isin(selected)]

# KPI section
st.header("1. Customer Overview")

total = len(df)
largest = df["Cluster"].value_counts().idxmax()
avg_spending = df[spending].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", total)
c2.metric("Total Clusters", k)
c3.metric("Largest Cluster", largest)
c4.metric("Average Spending", round(avg_spending, 2))

# Cluster distribution
st.header("2. Cluster Distribution")

counts = df["Cluster"].value_counts().sort_index()

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Customers in Each Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots()
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%"
    )
    ax.set_title("Customer Percentage")
    st.pyplot(fig)
    plt.close(fig)

# Cluster scatter plot
st.header("3. Cluster Visualization")

fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=filtered,
    x=income,
    y=spending,
    hue="Cluster",
    palette="tab10",
    ax=ax
)

# Cluster centers
centers = df.groupby("Cluster")[[income, spending]].mean()
centers = centers.loc[centers.index.isin(selected)]

ax.scatter(
    centers[income],
    centers[spending],
    color="black",
    marker="X",
    s=200,
    label="Centroids"
)

ax.set_title("Income vs Spending Score")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# Cluster profile
st.header("4. Cluster Profile")

profile_features = [income, spending]

if "Age" in df.columns:
    profile_features.append("Age")

profile = df.groupby("Cluster")[profile_features].mean()

profile["Customers"] = counts

st.dataframe(profile.round(2))

# Heatmap
st.subheader("Cluster Profile Heatmap")

heatmap_data = profile[profile_features]
heatmap_scaled = StandardScaler().fit_transform(heatmap_data)

fig, ax = plt.subplots(figsize=(8, 4))

sns.heatmap(
    heatmap_scaled,
    annot=True,
    cmap="coolwarm",
    xticklabels=profile_features,
    yticklabels=profile.index,
    ax=ax
)

ax.set_title("Cluster Comparison")
st.pyplot(fig)
plt.close(fig)

# PCA visualization
st.header("5. PCA Visualization")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(x_scaled)

df["PCA1"] = pca_data[:, 0]
df["PCA2"] = pca_data[:, 1]

pca_filtered = df[df["Cluster"].isin(selected)].copy()
pca_filtered["Cluster"] = pca_filtered["Cluster"].astype(str)

fig = px.scatter(
    pca_filtered,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    title="PCA Cluster Visualization"
)

st.plotly_chart(fig, use_container_width=True)

# Interactive chart
st.header("6. Interactive Customer Chart")

features = [income, spending]

if "Age" in df.columns:
    features.append("Age")

axis1 = st.selectbox("Select X Axis", features, index=0)
axis2 = st.selectbox("Select Y Axis", features, index=1)

plot_data = filtered.copy()
plot_data["Cluster"] = plot_data["Cluster"].astype(str)

hover_info = ["Cluster"]

if "CustomerID" in plot_data.columns:
    hover_info.append("CustomerID")

fig = px.scatter(
    plot_data,
    x=axis1,
    y=axis2,
    color="Cluster",
    hover_data=hover_info,
    title="Interactive Cluster Analysis"
)

st.plotly_chart(fig, use_container_width=True)

# Cluster comparison
st.header("7. Cluster Comparison")

st.dataframe(profile.loc[profile.index.isin(selected)].round(2))

# Customer records
st.header("8. Customer Records")

st.dataframe(filtered)

csv = filtered.to_csv(index=False)

st.download_button(
    "Download Customer Data",
    csv,
    "filtered_customers.csv",
    "text/csv"
)

# Business insights
st.header("9. Business Insights")

high_spending = profile[spending].idxmax()
low_spending = profile[spending].idxmin()
high_income = profile[income].idxmax()
smallest = counts.idxmin()

st.write(
    "1. Cluster", largest,
    "has the largest number of customers:",
    counts[largest]
)

st.write(
    "2. Cluster", high_spending,
    "has the highest average spending score:",
    round(profile.loc[high_spending, spending], 2),
    ". A loyalty program could be tested."
)

st.write(
    "3. Cluster", low_spending,
    "has the lowest average spending score:",
    round(profile.loc[low_spending, spending], 2),
    ". The business could investigate possible reasons."
)

st.write(
    "4. Cluster", high_income,
    "has the highest average annual income:",
    round(profile.loc[high_income, income], 2),
    "k. Premium products could be explored."
)

st.write(
    "5. Cluster", smallest,
    "has the fewest customers:",
    counts[smallest],
    ". A targeted campaign could be tested."
)

st.info("Business suggestions should be tested before use.")

# Save results
st.header("10. Save Results")

if st.button("Save Results"):
    import os

    os.makedirs("Day67_results", exist_ok=True)

    df.to_csv("Day67_results/clustered_customers.csv", index=False)
    profile.to_csv("Day67_results/cluster_profile.csv")
    counts.to_csv("Day67_results/cluster_sizes.csv")

    st.success("Results saved successfully!")

st.success("Customer Cluster Dashboard Completed!")
