import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Create output folder
output_folder = "Day72_results"
os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# PART 1 - DATA PREPARATION
# --------------------------------------------------

print("\nHYPERPARAMETER OPTIMIZATION SYSTEM")

# Load dataset
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

print("\nFirst 10 Records:")
print(df.head(10))

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nClass Distribution:")
print(df["Target"].value_counts())

# Separate features and target
X = df.drop("Target", axis=1)
y = df["Target"]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# --------------------------------------------------
# PART 2 - BUILD BASELINE MODELS
# --------------------------------------------------

baseline_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),

    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}

baseline_results = []

print("\nBASELINE MODEL RESULTS")

for name, model in baseline_models.items():

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    # Cross validation
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    baseline_results.append({
        "Model": name,
        "Type": "Baseline",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "CV Score": cv_scores.mean(),
        "Training Time": training_time,
        "Tuning Time": 0
    })

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("CV Score:", round(cv_scores.mean(), 4))
    print("Training Time:", round(training_time, 4), "seconds")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=data.target_names
    )

    display.plot(cmap="Blues")
    plt.title(name + " - Baseline Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_folder, name.replace(" ", "_") + "_baseline_cm.png")
    )

    plt.close()

# --------------------------------------------------
# PART 3 - HYPERPARAMETER TUNING
# --------------------------------------------------

tuned_models = {}

tuned_results = []

print("\nHYPERPARAMETER TUNING")

# 1. Decision Tree - GridSearchCV

print("\nTuning Decision Tree...")

dt_param_grid = {
    "max_depth": [None, 3, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

start_time = time.time()
dt_grid.fit(X_train, y_train)
dt_tuning_time = time.time() - start_time

tuned_models["Decision Tree"] = dt_grid.best_estimator_

print("Best Parameters:", dt_grid.best_params_)
print("Best CV Score:", round(dt_grid.best_score_, 4))

# 2. Random Forest - RandomizedSearchCV

print("\nTuning Random Forest...")

rf_param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
rf_random.fit(X_train, y_train)
rf_tuning_time = time.time() - start_time

tuned_models["Random Forest"] = rf_random.best_estimator_

print("Best Parameters:", rf_random.best_params_)
print("Best CV Score:", round(rf_random.best_score_, 4))

# 3. KNN - GridSearchCV with Pipeline

print("\nTuning KNN...")

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

knn_param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"]
}

knn_grid = GridSearchCV(
    knn_pipeline,
    knn_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

start_time = time.time()
knn_grid.fit(X_train, y_train)
knn_tuning_time = time.time() - start_time

tuned_models["KNN"] = knn_grid.best_estimator_

print("Best Parameters:", knn_grid.best_params_)
print("Best CV Score:", round(knn_grid.best_score_, 4))

# 4. Logistic Regression - GridSearchCV

print("\nTuning Logistic Regression...")

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

lr_param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"]
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

start_time = time.time()
lr_grid.fit(X_train, y_train)
lr_tuning_time = time.time() - start_time

tuned_models["Logistic Regression"] = lr_grid.best_estimator_

print("Best Parameters:", lr_grid.best_params_)
print("Best CV Score:", round(lr_grid.best_score_, 4))

# Store tuning times and CV scores
tuning_information = {
    "Decision Tree": (dt_tuning_time, dt_grid.best_score_),
    "Random Forest": (rf_tuning_time, rf_random.best_score_),
    "KNN": (knn_tuning_time, knn_grid.best_score_),
    "Logistic Regression": (lr_tuning_time, lr_grid.best_score_)
}

# --------------------------------------------------
# PART 4 - EVALUATE TUNED MODELS
# --------------------------------------------------

print("\nTUNED MODEL RESULTS")

for name, model in tuned_models.items():

    tuning_time, cv_score = tuning_information[name]

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tuned_results.append({
        "Model": name,
        "Type": "Tuned",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "CV Score": cv_score,
        "Training Time": training_time,
        "Tuning Time": tuning_time
    })

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("CV Score:", round(cv_score, 4))
    print("Training Time:", round(training_time, 4), "seconds")
    print("Tuning Time:", round(tuning_time, 4), "seconds")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=data.target_names
    )

    display.plot(cmap="Greens")
    plt.title(name + " - Tuned Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_folder, name.replace(" ", "_") + "_tuned_cm.png")
    )

    plt.close()

# --------------------------------------------------
# PART 5 - MODEL COMPARISON
# --------------------------------------------------

results = pd.DataFrame(baseline_results + tuned_results)

print("\nCOMPLETE MODEL COMPARISON")
print(results.round(4).to_string(index=False))

# Save results table
results.to_csv(
    os.path.join(output_folder, "model_comparison.csv"),
    index=False
)

# Accuracy comparison chart
accuracy_table = results.pivot(
    index="Model",
    columns="Type",
    values="Accuracy"
)

accuracy_table.plot(kind="bar", figsize=(10, 6))

plt.title("Baseline vs Tuned Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Machine Learning Models")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "accuracy_comparison.png"))
plt.close()

# Precision and Recall comparison
pr_table = results.melt(
    id_vars=["Model", "Type"],
    value_vars=["Precision", "Recall"],
    var_name="Metric",
    value_name="Score"
)

pr_table.pivot(
    index=["Model", "Type"],
    columns="Metric",
    values="Score"
).plot(kind="bar", figsize=(12, 6))

plt.title("Precision and Recall Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "precision_recall_comparison.png"))
plt.close()

# F1-score comparison
f1_table = results.pivot(
    index="Model",
    columns="Type",
    values="F1 Score"
)

f1_table.plot(kind="bar", figsize=(10, 6))

plt.title("Baseline vs Tuned F1 Score")
plt.ylabel("F1 Score")
plt.xlabel("Machine Learning Models")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "f1_comparison.png"))
plt.close()

# --------------------------------------------------
# PART 6 - FINAL REPORT
# --------------------------------------------------

best_index = results["F1 Score"].idxmax()
best_model = results.loc[best_index]

print("\nFINAL PROJECT REPORT")

print("\nBest Model:", best_model["Model"])
print("Model Type:", best_model["Type"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("Precision:", round(best_model["Precision"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("F1 Score:", round(best_model["F1 Score"], 4))

print("\nTUNING METHODS USED:")
print("Decision Tree - GridSearchCV")
print("Random Forest - RandomizedSearchCV")
print("KNN - GridSearchCV with Pipeline")
print("Logistic Regression - GridSearchCV with Pipeline")

print("\nBest Hyperparameters:")

print("Decision Tree:", dt_grid.best_params_)
print("Random Forest:", rf_random.best_params_)
print("KNN:", knn_grid.best_params_)
print("Logistic Regression:", lr_grid.best_params_)

print("\nDid tuning improve the results?")

for name in baseline_models.keys():

    before = results[
        (results["Model"] == name) &
        (results["Type"] == "Baseline")
    ]["Accuracy"].values[0]

    after = results[
        (results["Model"] == name) &
        (results["Type"] == "Tuned")
    ]["Accuracy"].values[0]

    if after > before:
        print(name, "- Accuracy improved")
    elif after < before:
        print(name, "- Accuracy decreased")
    else:
        print(name, "- Accuracy remained the same")

print("\nAll results saved successfully!")
print("Output Folder:", os.path.abspath(output_folder))
