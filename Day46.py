
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load customer purchase dataset
file_path = Path("customer_purchase.csv")

if file_path.exists():
    df = pd.read_csv(file_path)
    print("Existing dataset loaded successfully.")
else:
    np.random.seed(45)

    df = pd.DataFrame({
        "Age": np.random.randint(18, 70, 1000),
        "Income": np.random.randint(25000, 200000, 1000),
        "Website_Visits": np.random.randint(1, 30, 1000),
        "Time_Spent": np.random.randint(1, 120, 1000),
        "Previous_Purchases": np.random.randint(0, 20, 1000),
        "Product_Views": np.random.randint(1, 50, 1000)
    })

    df["Purchased"] = (
        (df["Website_Visits"] > 15) |
        (df["Time_Spent"] > 60) |
        (df["Previous_Purchases"] > 10)
    ).astype(int)

    df.to_csv(file_path, index=False)
    print("New customer purchase dataset created.")

print("\nFIRST 10 RECORDS")
print(df.head(10))

# Prepare features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Different tree depths
depths = [1, 2, 3, 5, 8, 12, 15]
train_accuracies = []
test_accuracies = []
differences = []

print("\nDECISION TREE DEPTH COMPARISON")
print("-" * 65)
print(f"{'Depth':<12}{'Train Accuracy':<18}{'Test Accuracy':<18}{'Difference':<12}")

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    difference = train_acc - test_acc

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    differences.append(difference)

    print(f"{depth:<12}{train_acc*100:<18.2f}{test_acc*100:<18.2f}{difference*100:<12.2f}")

# Accuracy comparison graph
plt.figure(figsize=(10, 6))

plt.plot(depths, train_accuracies,
         marker="o", label="Training Accuracy")

plt.plot(depths, test_accuracies,
         marker="s", label="Testing Accuracy")

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting Analysis")
plt.xticks(depths)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Find best tree depth
best_index = np.argmax(test_accuracies)
best_depth = depths[best_index]

print("\nFINAL ANALYSIS")
print("-" * 45)

print("Best Tree Depth:", best_depth)
print(f"Best Testing Accuracy: {test_accuracies[best_index]*100:.2f}%")

print("\nUnderfitting:")
print("Small tree depths may fail to learn enough patterns.")

print("\nBalanced Model:")
print(f"Depth {best_depth} gives the highest testing accuracy.")

print("\nOverfitting:")
print("Large training-testing accuracy gaps suggest overfitting.")

# Bonus: Unrestricted Decision Tree
unrestricted_model = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

unrestricted_model.fit(X_train, y_train)

unrestricted_train = accuracy_score(
    y_train, unrestricted_model.predict(X_train)
)

unrestricted_test = accuracy_score(
    y_test, unrestricted_model.predict(X_test)
)

print("\nBONUS: UNRESTRICTED TREE")
print(f"Training Accuracy: {unrestricted_train*100:.2f}%")
print(f"Testing Accuracy: {unrestricted_test*100:.2f}%")
print("Actual Tree Depth:", unrestricted_model.get_depth())

print("\nCONCLUSION")
print("Increasing tree depth can improve training accuracy.")
print("However, very deep trees may overfit the training data.")
print("The best depth should be selected using unseen data performance.")
