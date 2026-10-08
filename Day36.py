
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
sns.set_theme(style="whitegrid")

os.makedirs("Day36_Output", exist_ok=True)
output = "Day36_Output"

# PART 1: DATASET GENERATION

print("\nREAL ESTATE PRICE PREDICTION SYSTEM")

file = "house_data_day36.csv"

if not os.path.exists(file):
    n = 2500

    area = np.random.randint(500, 5000, n)
    bedrooms = np.random.randint(1, 7, n)
    bathrooms = np.random.randint(1, 5, n)
    floors = np.random.randint(1, 4, n)
    age = np.random.randint(0, 41, n)
    parking = np.random.randint(0, 4, n)
    distance = np.random.uniform(1, 40, n).round(2)

    location = np.random.choice(
        ["Urban", "Suburban", "Rural"], n
    )

    property_type = np.random.choice(
        ["House", "Apartment", "Villa"], n
    )

    location_bonus = pd.Series(location).map({
        "Urban": 80000,
        "Suburban": 40000,
        "Rural": 10000
    }).to_numpy()

    type_bonus = pd.Series(property_type).map({
        "House": 20000,
        "Apartment": 10000,
        "Villa": 100000
    }).to_numpy()

    price = (
        area * 150
        + bedrooms * 10000
        + bathrooms * 15000
        + floors * 12000
        - age * 2000
        + parking * 8000
        - distance * 1500
        + location_bonus
        + type_bonus
        + np.random.normal(0, 25000, n)
    )

    price = np.maximum(price, 50000).round(2)

    data = pd.DataFrame({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Property_Age": age,
        "Location": location,
        "Parking_Spaces": parking,
        "Distance_to_City": distance,
        "Property_Type": property_type,
        "Price": price
    })

    data.to_csv(file, index=False)
    print("\nDataset generated successfully.")

df = pd.read_csv(file)

print("\nFIRST 10 RECORDS")
print(df.head(10))

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all"))

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATES:", df.duplicated().sum())

# DATA CLEANING

df = df.drop_duplicates().copy()

numeric_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Property_Age",
    "Parking_Spaces",
    "Distance_to_City"
]

categorical_features = [
    "Location",
    "Property_Type"
]

for col in numeric_features + ["Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in numeric_features:
    df.loc[df[col] < 0, col] = np.nan

for col in ["Area", "Bedrooms", "Bathrooms", "Floors"]:
    df.loc[df[col] <= 0, col] = np.nan

df.loc[df["Price"] <= 0, "Price"] = np.nan

for col in categorical_features:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

df = df.dropna(subset=["Price"])

print("\nCLEANED DATASET SIZE:", len(df))

df.to_csv(
    f"{output}/cleaned_house_data.csv",
    index=False
)

# PART 2: EXPLORATORY DATA ANALYSIS

print("\nEXPLORATORY DATA ANALYSIS")

plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=35, kde=True)
plt.title("House Price Distribution")
plt.tight_layout()
plt.savefig(f"{output}/price_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Area", y="Price")
plt.title("Area vs Price")
plt.tight_layout()
plt.savefig(f"{output}/area_vs_price.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Bedrooms", y="Price")
plt.title("Bedrooms vs Price")
plt.tight_layout()
plt.savefig(f"{output}/bedrooms_vs_price.png")
plt.show()

df[numeric_features].hist(figsize=(13, 10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig(f"{output}/feature_distributions.png")
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(
    df[numeric_features + ["Price"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output}/correlation_heatmap.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Price"])
plt.title("House Price Outlier Analysis")
plt.tight_layout()
plt.savefig(f"{output}/outlier_analysis.png")
plt.show()

# PART 3: MODEL TRAINING

print("\nMODEL TRAINING")

features = numeric_features + categorical_features

X = df[features]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])

model.fit(X_train, y_train)

print("Training completed.")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# PART 4: PREDICTION

print("\nTEST DATA PREDICTIONS")

predictions = model.predict(X_test)

results = X_test.copy()
results["Actual_Price"] = y_test
results["Predicted_Price"] = predictions
results["Error"] = (
    results["Actual_Price"]
    - results["Predicted_Price"]
)
results["Absolute_Error"] = results["Error"].abs()

print(results.head(10).to_string())

results.to_csv(
    f"{output}/test_predictions.csv",
    index=False
)

# PREDICT 10 NEW PROPERTIES

new_properties = pd.DataFrame({
    "Area": [
        900, 1200, 1500, 1800, 2100,
        2400, 2700, 3000, 3500, 4200
    ],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 4, 5, 5, 6],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    "Floors": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    "Property_Age": [20, 15, 10, 8, 5, 12, 4, 3, 2, 1],
    "Location": [
        "Rural", "Suburban", "Urban",
        "Suburban", "Urban", "Rural",
        "Urban", "Suburban", "Urban", "Urban"
    ],
    "Parking_Spaces": [0, 1, 1, 1, 2, 2, 2, 2, 3, 3],
    "Distance_to_City": [
        35, 20, 10, 15, 5,
        30, 8, 12, 4, 2
    ],
    "Property_Type": [
        "House", "Apartment", "House",
        "House", "Villa", "House",
        "Apartment", "Villa", "Villa", "Villa"
    ]
})

new_properties["Predicted_Price"] = model.predict(
    new_properties
)

print("\n10 NEW PROPERTY PREDICTIONS")
print(new_properties.to_string(index=False))

new_properties.to_csv(
    f"{output}/new_property_predictions.csv",
    index=False
)

# PART 5: MODEL EVALUATION

print("\nMODEL EVALUATION")

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R2   : {r2:.4f}")

# ACTUAL VS PREDICTED

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=y_test,
    y=predictions,
    alpha=0.6
)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.savefig(f"{output}/actual_vs_predicted.png")
plt.show()

# RESIDUAL DISTRIBUTION

plt.figure(figsize=(8, 5))
sns.histplot(
    results["Error"],
    bins=35,
    kde=True
)
plt.title("Residual Distribution")
plt.xlabel("Prediction Error")
plt.tight_layout()
plt.savefig(f"{output}/residual_distribution.png")
plt.show()

# PREDICTION ERRORS

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=results["Predicted_Price"],
    y=results["Error"]
)

plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Prediction Error Analysis")
plt.tight_layout()
plt.savefig(f"{output}/prediction_errors.png")
plt.show()

# PART 6: ERROR ANALYSIS

print("\nERROR ANALYSIS")

largest_errors = results.nlargest(
    10,
    "Absolute_Error"
)

smallest_errors = results.nsmallest(
    10,
    "Absolute_Error"
)

print("\n10 LARGEST ERRORS")
print(largest_errors.to_string())

print("\n10 SMALLEST ERRORS")
print(smallest_errors.to_string())

largest_errors.to_csv(
    f"{output}/largest_errors.csv",
    index=False
)

smallest_errors.to_csv(
    f"{output}/smallest_errors.csv",
    index=False
)

# DETECT POSSIBLE OUTLIERS

q1 = results["Absolute_Error"].quantile(0.25)
q3 = results["Absolute_Error"].quantile(0.75)

iqr = q3 - q1
threshold = q3 + 1.5 * iqr

outliers = results[
    results["Absolute_Error"] > threshold
]

print("\nPOSSIBLE ERROR OUTLIERS")
print(outliers.to_string())

print("\nNUMBER OF OUTLIERS:", len(outliers))

outliers.to_csv(
    f"{output}/possible_outliers.csv",
    index=False
)

# IDENTIFY POOR PERFORMANCE AREAS

print("\nERROR BY LOCATION")

location_errors = results.groupby(
    "Location"
)["Absolute_Error"].agg(
    ["count", "mean", "max"]
)

print(location_errors)

print("\nERROR BY PROPERTY TYPE")

type_errors = results.groupby(
    "Property_Type"
)["Absolute_Error"].agg(
    ["count", "mean", "max"]
)

print(type_errors)

location_errors.to_csv(
    f"{output}/location_errors.csv"
)

type_errors.to_csv(
    f"{output}/property_type_errors.csv"
)

# PART 7: FINAL REPORT

best = results.loc[
    results["Absolute_Error"].idxmin()
]

worst = results.loc[
    results["Absolute_Error"].idxmax()
]

if r2 >= 0.90:
    conclusion = "Strong predictive performance."
elif r2 >= 0.75:
    conclusion = "Reasonable predictive performance."
else:
    conclusion = "Model requires improvement."

report = f"""
REAL ESTATE PRICE PREDICTION
FINAL PROJECT REPORT

Dataset Size       : {len(df)}
Features Used      : {", ".join(features)}
Training Records   : {len(X_train)}
Testing Records    : {len(X_test)}

MAE                : {mae:,.2f}
MSE                : {mse:,.2f}
RMSE               : {rmse:,.2f}
R2                 : {r2:.4f}

Best Prediction
Actual Price       : {best["Actual_Price"]:,.2f}
Predicted Price    : {best["Predicted_Price"]:,.2f}
Absolute Error     : {best["Absolute_Error"]:,.2f}

Worst Prediction
Actual Price       : {worst["Actual_Price"]:,.2f}
Predicted Price    : {worst["Predicted_Price"]:,.2f}
Absolute Error     : {worst["Absolute_Error"]:,.2f}

Largest Error      : {results["Absolute_Error"].max():,.2f}
Possible Outliers  : {len(outliers)}

Model Conclusion:
{conclusion}

Note:
This project uses synthetic property data.
Results do not establish real-world accuracy.

Recommended Improvements:
1. Use real-world property data.
2. Compare different feature combinations.
3. Test alternative regression models.
4. Perform cross-validation.
5. Investigate outliers and residual patterns.
"""

print(report)

with open(
    f"{output}/final_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# BONUS: SAVE TRAINED MODEL

joblib.dump(
    model,
    f"{output}/house_price_model.pkl"
)

print("\nModel saved successfully.")
print("All results saved in:", output)
print("\nPROJECT COMPLETED SUCCESSFULLY!")
