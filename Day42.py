import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create dataset
np.random.seed(42)

n = 2000

data = {
    "Customer_ID": range(1, n + 1),
    "Age": np.random.randint(18, 70, n),
    "Tenure": np.random.randint(1, 73, n),
    "Monthly_Charges": np.round(np.random.uniform(20, 150, n), 2),
    "Support_Calls": np.random.randint(0, 10, n),
    "Contract_Type": np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"], n
    ),
    "Payment_Method": np.random.choice(
        ["Credit Card", "Bank Transfer", "Electronic Check", "Cash"], n
    ),
    "Internet_Usage": np.round(np.random.uniform(1, 500, n), 2),
    "Previous_Complaints": np.random.randint(0, 6, n)
}

df = pd.DataFrame(data)

# Calculate Total Charges
df["Total_Charges"] = (
    df["Monthly_Charges"] * df["Tenure"]
).round(2)

# Create realistic churn target
churn_score = (
    (df["Tenure"] < 12).astype(int)
    + (df["Monthly_Charges"] > 100).astype(int)
    + (df["Support_Calls"] >= 5).astype(int)
    + (df["Previous_Complaints"] >= 3).astype(int)
    + (df["Contract_Type"] == "Month-to-Month").astype(int)
)

df["Churn"] = (churn_score >= 3).astype(int)

# Add a few missing values for cleaning practice
df.loc[10, "Monthly_Charges"] = np.nan
df.loc[25, "Internet_Usage"] = np.nan
df.loc[50, "Total_Charges"] = np.nan

# Save dataset
df.to_csv("customer_churn.csv", index=False)


# Load dataset
df = pd.read_csv("customer_churn.csv")

print("\nFirst 10 Records:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# Handle missing values
numeric_columns = [
    "Age",
    "Tenure",
    "Monthly_Charges",
    "Total_Charges",
    "Support_Calls",
    "Internet_Usage",
    "Previous_Complaints"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Remove duplicates
df = df.drop_duplicates()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())


# Exploratory Data Analysis

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Churn Distribution")
plt.show()

# 2. Churn by Contract Type
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Contract_Type", hue="Churn")
plt.title("Churn by Contract Type")
plt.show()

# 3. Monthly Charges vs Churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="Monthly_Charges")
plt.title("Monthly Charges vs Churn")
plt.show()

# 4. Tenure vs Churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="Tenure")
plt.title("Tenure vs Churn")
plt.show()

# 5. Support Calls vs Churn
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Support_Calls", hue="Churn")
plt.title("Support Calls vs Churn")
plt.show()


# Encode categorical columns
encoder = LabelEncoder()

categorical_columns = [
    "Contract_Type",
    "Payment_Method"
]

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])


# Prepare X and y
X = df.drop(["Customer_ID", "Churn"], axis=1)
y = df["Churn"]

# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# Train Logistic Regression
model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)

# Churn probability
y_probability = model.predict_proba(X_test)[:, 1]


# Prediction Results
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Churn_Probability": np.round(y_probability, 4)
})

print("\nFirst 20 Predictions:")
print(results.head(20))


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

correct_predictions = (y_test.values == y_pred).sum()
incorrect_predictions = (y_test.values != y_pred).sum()

print("\nMODEL EVALUATION")
print("-----------------------------")

print("Total Test Records:", len(y_test))
print("Correct Predictions:", correct_predictions)
print("Incorrect Predictions:", incorrect_predictions)
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nTrue Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Not Churn", "Churn"],
    yticklabels=["Not Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# Error Analysis
error_results = X_test.copy()

error_results["Actual"] = y_test
error_results["Predicted"] = y_pred
error_results["Probability"] = y_probability

incorrect = error_results[
    error_results["Actual"] != error_results["Predicted"]
]

print("\nIncorrectly Classified Customers:")
print(incorrect.head(20))

print("\nNumber of False Positives:", fp)
print("Number of False Negatives:", fn)

if fp > fn:
    print("False Positives are more common.")
elif fn > fp:
    print("False Negatives are more common.")
else:
    print("False Positives and False Negatives are equal.")


# Check dataset balance
print("\nClass Distribution:")
print(y.value_counts())

print("\nClass Distribution Percentage:")
print(y.value_counts(normalize=True) * 100)


# Predict 10 new customers
new_customers = X.sample(10, random_state=10)

new_predictions = model.predict(new_customers)
new_probabilities = model.predict_proba(new_customers)[:, 1]

new_results = new_customers.copy()

new_results["Predicted_Churn"] = new_predictions
new_results["Churn_Probability"] = np.round(
    new_probabilities, 4
)

print("\nPredictions for 10 New Customers:")
print(new_results)


# Final Business Recommendation
print("\nFINAL BUSINESS RECOMMENDATION")
print("-----------------------------")

if accuracy >= 0.90:
    performance = "Good"
elif accuracy >= 0.75:
    performance = "Moderate"
else:
    performance = "Poor"

print("Model Performance:", performance)

if accuracy >= 0.80:
    print("Can the company use this model? Yes, with improvements.")
else:
    print("Can the company use this model? No, further improvement is needed.")

print(
    "Reason: The model can identify customers who may leave "
    "based on their customer information."
)

print(
    "Recommended Improvements: Collect more real customer data, "
    "improve feature selection, handle class imbalance, and compare "
    "Logistic Regression with other classification models."
)