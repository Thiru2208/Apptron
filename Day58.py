
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder
output_dir = Path(__file__).resolve().parent / "Day58_results"
output_dir.mkdir(exist_ok=True)

# Generate synthetic customer dataset
np.random.seed(42)

n = 250

data = {
    "Customer_ID": np.arange(1, n + 1),
    "Age": np.random.randint(18, 71, n),
    "Annual_Income": np.random.randint(15, 151, n),
    "Spending_Score": np.random.randint(1, 101, n)
}

df = pd.DataFrame(data)

# Save original dataset
df.to_csv(output_dir / "customer_data.csv", index=False)

# Load dataset using Pandas
df = pd.read_csv(output_dir / "customer_data.csv")

print("\nCUSTOMER DATA EXPLORATION SYSTEM")

# Display first 10 records
print("\nFirst 10 Records:")
print(df.head(10))

# Check rows and columns
print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# Inspect data types
print("\nData Types:")
print(df.dtypes)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

print("\nDuplicate Customer IDs:",
      df["Customer_ID"].duplicated().sum())

# Clean dataset
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["Customer_ID"])
df = df.dropna().copy()

# Summary statistics
features = ["Age", "Annual_Income", "Spending_Score"]

summary = df[features].agg(
    ["mean", "median", "min", "max"]
)

print("\nSummary Statistics:")
print(summary)

summary.to_csv(output_dir / "summary_statistics.csv")

# Create three histograms
for feature in features:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[feature],
        bins=15,
        color="skyblue",
        edgecolor="black"
    )

    plt.title(f"Distribution of {feature.replace('_', ' ')}")
    plt.xlabel(feature.replace("_", " "))
    plt.ylabel("Number of Customers")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{feature}_histogram.png",
        dpi=300
    )

    plt.show()
    plt.close()

# Scatter plot
plt.figure(figsize=(9, 6))

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    color="blue",
    alpha=0.6
)

plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (Thousands)")
plt.ylabel("Spending Score (1-100)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "income_vs_spending.png",
    dpi=300
)

plt.show()
plt.close()

# Identify customer patterns
high_income_high_spending = df[
    (df["Annual_Income"] >= 100) &
    (df["Spending_Score"] >= 70)
]

high_income_low_spending = df[
    (df["Annual_Income"] >= 100) &
    (df["Spending_Score"] <= 30)
]

low_income_high_spending = df[
    (df["Annual_Income"] <= 40) &
    (df["Spending_Score"] >= 70)
]

print("\nCUSTOMER PATTERN ANALYSIS")

print("\n1. High Income and High Spending:",
      len(high_income_high_spending))

print("2. High Income and Low Spending:",
      len(high_income_low_spending))

print("3. Low Income and High Spending:",
      len(low_income_high_spending))

# Detect unusual records using IQR
Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)

IQR = Q3 - Q1

outlier_mask = (
    (df[features] < (Q1 - 1.5 * IQR)) |
    (df[features] > (Q3 + 1.5 * IQR))
).any(axis=1)

outliers = df[outlier_mask]

print("\nNumber of Unusual Records:", len(outliers))

outliers.to_csv(
    output_dir / "unusual_records.csv",
    index=False
)

# Select suitable features for clustering
clustering_features = df[
    ["Age", "Annual_Income", "Spending_Score"]
].copy()

print("\nSelected Clustering Features:")
print(clustering_features.head(10))

clustering_features.to_csv(
    output_dir / "clustering_features.csv",
    index=False
)

# Save cleaned dataset
df.to_csv(
    output_dir / "cleaned_customer_data.csv",
    index=False
)

# Generate observation report
report = f"""
DAY 58 - CUSTOMER DATA EXPLORATION REPORT

Dataset Information
-------------------
Total Customers: {len(df)}
Total Columns: {df.shape[1]}
Missing Values: {df.isnull().sum().sum()}
Duplicate Records: {df.duplicated().sum()}

Summary Statistics
------------------
{summary.to_string()}

Customer Patterns
-----------------
1. High income and high spending customers:
   {len(high_income_high_spending)}

2. High income and low spending customers:
   {len(high_income_low_spending)}

3. Low income and high spending customers:
   {len(low_income_high_spending)}

Unusual Records
---------------
Number of IQR outliers: {len(outliers)}

Selected Clustering Features
----------------------------
Age
Annual Income
Spending Score

Customer ID is excluded because it is only
a unique identifier and does not represent
customer behavior or similarity.

Conclusion
----------
The dataset was explored and cleaned successfully.

Three histograms and one scatter plot were created.

The selected numerical features are suitable
for further clustering analysis after scaling.

Note:
Synthetic data was used for this exercise.
The customer patterns are illustrative and
may not represent real customer behavior.
"""

print("\n" + report)

with open(
    output_dir / "observation_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\nAll results saved successfully!")
print("Output Folder:", output_dir)
