import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(45)

# Create 1000 customer records
data = {"Age": np.random.randint(18, 70, 1000),
        "Income": np.random.randint(25000, 200000, 1000),
        "Website_Visits": np.random.randint(1, 30, 1000),
        "Time_Spent": np.random.randint(1, 120, 1000),
        "Previous_Purchases": np.random.randint(0, 20, 1000),
        "Product_Views": np.random.randint(1, 50, 1000)
        }

df = pd.DataFrame(data)

# Create target column
df["Purchased"] = ( 
    (df["Website_Visits"] > 15 ) |
    (df["Time_Spent"] > 60) |
    (df["Previous_Purchases"] > 10)
).astype(int)

# Save dataset
df.to_csv("customer_purchase.csv", index = False)

#Load CSV
df = pd.read_csv("customer_purchase.csv")

print("First 10 Records")
print(df.head(10))

print("\nDataset Information")
print(df.info())

print("\nClass Distribution")
print(df["Purchased"].value_counts())

# Features
X =df.drop("Purchased", axis=1)

# Target
y = df["Purchased"]

print("\nFeatures")
print(X.columns.to_list())

print("\nTarget")
print("Purchased")

# 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split( X, y,
                                                    test_size= 0.20,
                                                    random_state = 42,
                                                    stratify = y)

print("\nDATASET SUMMARY")
print("Total Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Number of Features:", X.shape[1])
print("Class 0:", (y == 0).sum())
print("Class 1:", (y == 1).sum())

print("\nTRAINING CLASS DISTRIBUTION")
print(Y_train.value_counts())

print("\nTESTING CLASS DISTRIBUTION")
print(Y_test.value_counts())