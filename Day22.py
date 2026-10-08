import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "raw_customer_data.csv"
CLEAN_FILE = BASE_DIR / "clean_customer_data.csv"
ML_FILE = BASE_DIR / "ml_ready_customer_data.csv"

 
# CREATE CUSTOMER DATASET 

def create_customer_dataset():

    np.random.seed(42)

    total_customers = 5200

    customer_id = np.arange(1, total_customers + 1)

    age = np.random.randint(18, 81, total_customers).astype(float)

    gender = np.random.choice(
        ["Male", "Female"],
        total_customers
    )

    city = np.random.choice(
        ["Colombo", "Kandy", "Galle", "Jaffna", "Kurunegala"],
        total_customers
    )

    tenure = np.random.randint(1, 73, total_customers)

    monthly_charges = np.random.uniform(
        1500, 15000, total_customers
    ).round(2)

    total_charges = (
        monthly_charges * tenure
    ).round(2)

    contract_type = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        total_customers
    )

    payment_method = np.random.choice(
        [
            "Credit Card",
            "Bank Transfer",
            "Cash",
            "Online Payment"
        ],
        total_customers
    )

    support_calls = np.random.randint(
        0, 11, total_customers
    )

    internet_usage = np.random.uniform(
        1, 500, total_customers
    ).round(2)

    churn = np.random.choice(
        ["Yes", "No"],
        total_customers,
        p=[0.30, 0.70]
    )

    data = pd.DataFrame({

        "Customer_ID": customer_id,
        "Age": age,
        "Gender": gender,
        "City": city,
        "Tenure": tenure,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "Contract_Type": contract_type,
        "Payment_Method": payment_method,
        "Support_Calls": support_calls,
        "Internet_Usage": internet_usage,
        "Churn": churn
    })


    # ADD SOME MISSING VALUES

    data.loc[
        np.random.choice(data.index, 50, replace=False),
        "Age"
    ] = np.nan

    data.loc[
        np.random.choice(data.index, 40, replace=False),
        "Monthly_Charges"
    ] = np.nan

    data.loc[
        np.random.choice(data.index, 40, replace=False),
        "Total_Charges"
    ] = np.nan

    # ADD INVALID AGES
     
    data.loc[10, "Age"] = -5
    data.loc[25, "Age"] = 150
    data.loc[50, "Age"] = -10
    data.loc[75, "Age"] = 200
     
    # ADD INVALID CHARGES
    
    data.loc[100, "Monthly_Charges"] = -500
    data.loc[150, "Monthly_Charges"] = -1000

    data.loc[200, "Total_Charges"] = -5000
    data.loc[250, "Total_Charges"] = -10000

    # ADD INCONSISTENT CATEGORICAL VALUES

    data.loc[20, "Gender"] = "male"
    data.loc[30, "Gender"] = "FEMALE"

    data.loc[40, "City"] = "colombo"
    data.loc[60, "City"] = "KANDY"

    data.loc[80, "Contract_Type"] = "one year"
    data.loc[90, "Contract_Type"] = "TWO YEAR"

    data.loc[110, "Payment_Method"] = "cash"
    data.loc[120, "Payment_Method"] = "CREDIT CARD"
     
    # ADD DUPLICATE RECORDS
     
    duplicates = data.iloc[0:50].copy()

    data = pd.concat(
        [data, duplicates],
        ignore_index=True
    )

    # Save raw dataset

    data.to_csv(
        RAW_FILE,
        index=False
    )

    print("\nRAW CUSTOMER DATASET CREATED")
    print("------------------------------------------")
    print("Total Records:", len(data))

    print("\nFirst 5 Records:")
    print(data.head())

    print("\nFile saved as:", RAW_FILE)

    return data

# LOAD DATASET 

def load_dataset():

    data = pd.read_csv(
        RAW_FILE
    )

    print("\n\nDATASET LOADED SUCCESSFULLY")
    print("------------------------------------------")

    return data

 
# INSPECT DATASET 

def inspect_dataset(data):

    print("\n\nDATASET INSPECTION")
    print("=" * 60)

    print("\n1. Shape of Dataset")
    print("------------------------------")
    print(data.shape)

    print("\n2. Column Names")
    print("------------------------------")
    print(data.columns.tolist())

    print("\n3. Data Types")
    print("------------------------------")
    print(data.dtypes)

    print("\n4. Missing Values")
    print("------------------------------")
    print(data.isnull().sum())

    print("\n5. Duplicate Count")
    print("------------------------------")
    print(data.duplicated().sum())

    print("\n6. Statistical Summary")
    print("------------------------------")
    print(data.describe())

 
# CLEAN DATASET 

def clean_dataset(data):

    print("\n\nDATA CLEANING")
    print("=" * 60)

    data = data.copy()

    missing_before = data.isnull().sum().sum()

    duplicate_before = data.duplicated().sum()


     
    # 1. REMOVE DUPLICATES
     

    data.drop_duplicates(
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )

    print("\nDuplicates Removed:",
          duplicate_before)


     
    # 2. CLEAN INVALID AGES
     

    # Valid age range = 18 to 100

    invalid_age = (
        (data["Age"] < 18) |
        (data["Age"] > 100)
    )

    print("Invalid Ages Found:",
          invalid_age.sum())

    data.loc[
        invalid_age,
        "Age"
    ] = np.nan


    # Replace missing/invalid age using median

    age_median = data["Age"].median()

    data["Age"] = data["Age"].fillna(
        age_median
    )


     
    # 3. CLEAN MONTHLY CHARGES
     

    invalid_monthly = (
        data["Monthly_Charges"] < 0
    )

    print("Invalid Monthly Charges Found:",
          invalid_monthly.sum())

    data.loc[
        invalid_monthly,
        "Monthly_Charges"
    ] = np.nan

    monthly_median = (
        data["Monthly_Charges"].median()
    )

    data["Monthly_Charges"] = (
        data["Monthly_Charges"]
        .fillna(monthly_median)
    )


     
    # 4. CLEAN TOTAL CHARGES
     

    invalid_total = (
        data["Total_Charges"] < 0
    )

    print("Invalid Total Charges Found:",
          invalid_total.sum())

    data.loc[
        invalid_total,
        "Total_Charges"
    ] = np.nan

    total_median = (
        data["Total_Charges"].median()
    )

    data["Total_Charges"] = (
        data["Total_Charges"]
        .fillna(total_median)
    )
     
    # 5. CONSISTENT CATEGORICAL VALUES
     

    data["Gender"] = (
        data["Gender"]
        .str.strip()
        .str.title()
    )

    data["City"] = (
        data["City"]
        .str.strip()
        .str.title()
    )

    data["Contract_Type"] = (
        data["Contract_Type"]
        .str.strip()
        .str.title()
    )

    data["Payment_Method"] = (
        data["Payment_Method"]
        .str.strip()
        .str.title()
    )

    data["Churn"] = (
        data["Churn"]
        .str.strip()
        .str.title()
    )
     
    # CHECK MISSING VALUES AFTER CLEANING
     

    missing_after = (
        data.isnull()
        .sum()
        .sum()
    )

    # SAVE CLEAN DATASET
     
    data.to_csv(
        CLEAN_FILE,
        index=False
    )

    print("\nMissing Values Before:",
          missing_before)

    print("Missing Values After:",
          missing_after)

    print("\nClean Dataset Shape:",
          data.shape)

    print("\nClean dataset saved as:")
    print(CLEAN_FILE)

    return (
        data,
        missing_before,
        missing_after,
        duplicate_before
    )

 
#  FILTER DATA 

def filter_customers(data):

    print("\n\nCUSTOMER FILTERING")
    print("=" * 60)


    # HIGH-VALUE CUSTOMERS
     
    high_value_customers = data[
        data["Total_Charges"] >= 500000
    ]

    print("\n High-Value Customers")
    print("------------------------------------------")
    print("Total:", len(high_value_customers))

    print(
        high_value_customers[
            [
                "Customer_ID",
                "Total_Charges",
                "Churn"
            ]
        ].head()
    )

    # CUSTOMERS WITH MANY SUPPORT CALLS
     

    many_support_calls = data[
        data["Support_Calls"] >= 7
    ]

    print("\n Customers With Many Support Calls")
    print("------------------------------------------")
    print("Total:", len(many_support_calls))

    print(
        many_support_calls[
            [
                "Customer_ID",
                "Support_Calls",
                "Churn"
            ]
        ].head()
    )
     
    # HIGH MONTHLY CHARGES
     

    high_monthly_charges = data[
        data["Monthly_Charges"] >= 10000
    ]

    print("\n Customers With High Monthly Charges")
    print("------------------------------------------")
    print("Total:", len(high_monthly_charges))

    print(
        high_monthly_charges[
            [
                "Customer_ID",
                "Monthly_Charges",
                "Churn"
            ]
        ].head()
    )
     
    # CHURNED CUSTOMERS
     

    churned_customers = data[
        data["Churn"] == "Yes"
    ]

    print("\n Customers Who Churned")
    print("------------------------------------------")
    print("Total:", len(churned_customers))

    print(
        churned_customers[
            [
                "Customer_ID",
                "Contract_Type",
                "Churn"
            ]
        ].head()
    )
     
    # CUSTOMERS WHO STAYED

    stayed_customers = data[
        data["Churn"] == "No"
    ]

    print("\n Customers Who Stayed")
    print("------------------------------------------")
    print("Total:", len(stayed_customers))

    print(
        stayed_customers[
            [
                "Customer_ID",
                "Contract_Type",
                "Churn"
            ]
        ].head()
    )

# GROUP AND ANALYZE 

def group_and_analyze(data):

    print("\n\nGROUP AND ANALYZE")
    print("=" * 60)

    # CHURN BY CONTRACT TYPE

    churn_contract = pd.crosstab(
        data["Contract_Type"],
        data["Churn"]
    )

    print("\n Churn by Contract Type")
    print("------------------------------------------")
    print(churn_contract)
     
    # CHURN BY CITY

    churn_city = pd.crosstab(
        data["City"],
        data["Churn"]
    )

    print("\n Churn by City")
    print("------------------------------------------")
    print(churn_city)

    # AVERAGE CHARGES BY CONTRACT TYPE
     

    average_charges = (
        data.groupby(
            "Contract_Type"
        )["Monthly_Charges"]
        .mean()
        .round(2)
    )

    print("\n Average Monthly Charges by Contract Type")
    print("------------------------------------------")
    print(average_charges)
     
    # AVERAGE SUPPORT CALLS BY CHURN STATUS
    
    average_support = (
        data.groupby(
            "Churn"
        )["Support_Calls"]
        .mean()
        .round(2)
    )

    print("\n Average Support Calls by Churn Status")
    print("------------------------------------------")
    print(average_support)

 
# PREPARE MACHINE LEARNING DATASET 

def prepare_ml_dataset(data):

    print("\n\nMACHINE LEARNING DATA PREPARATION")
    print("=" * 60)

    ml_data = data.copy()
     
    # REMOVE UNNECESSARY COLUMN
     

    ml_data.drop(
        columns=["Customer_ID"],
        inplace=True
    )
     
    # CONVERT TARGET
    # Yes = 1
    # No  = 0
     

    ml_data["Churn"] = (
        ml_data["Churn"]
        .map({
            "Yes": 1,
            "No": 0
        })
    )


    # CONVERT CATEGORICAL COLUMNS
     

    categorical_columns = [
        "Gender",
        "City",
        "Contract_Type",
        "Payment_Method"
    ]

    ml_data = pd.get_dummies(
        ml_data,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )
     
    # SEPARATE FEATURES AND TARGET
     

    X = ml_data.drop(
        columns=["Churn"]
    )

    y = ml_data["Churn"]


    print("\nFeatures (X)")
    print("------------------------------------------")
    print(X.head())

    print("\nTarget (y)")
    print("------------------------------------------")
    print(y.head())

    print("\nX Shape:", X.shape)
    print("y Shape:", y.shape)


    # SAVE ML READY DATA
     

    ml_data.to_csv(
        ML_FILE,
        index=False
    )

    print("\nML-ready dataset saved as:")
    print(ML_FILE)

    return X, y

 
# FINAL REPORT 

def final_report(
        raw_data,
        clean_data,
        removed_duplicates,
        missing_before,
        missing_after
):

    print("\n\nFINAL REPORT")
    print("=" * 60)

    total_customers = len(raw_data)

    clean_customers = len(clean_data)

    churned_customers = (
        clean_data[
            clean_data["Churn"] == "Yes"
        ].shape[0]
    )

    churn_rate = (
        churned_customers /
        clean_customers
    ) * 100

    average_monthly_charges = (
        clean_data[
            "Monthly_Charges"
        ].mean()
    )
     
    # FIND CHURN RATE BY CONTRACT TYPE
     
    churn_segment = (
        clean_data
        .assign(
            Churn_Number=(
                clean_data["Churn"] == "Yes"
            ).astype(int)
        )
        .groupby(
            "Contract_Type"
        )["Churn_Number"]
        .mean()
    )


    highest_churn_segment = (
        churn_segment.idxmax()
    )


    print(
        f"Total Customers: "
        f"{total_customers}"
    )

    print(
        f"Clean Customers: "
        f"{clean_customers}"
    )

    print(
        f"Removed Duplicates: "
        f"{removed_duplicates}"
    )

    print(
        f"Missing Values Before: "
        f"{missing_before}"
    )

    print(
        f"Missing Values After: "
        f"{missing_after}"
    )

    print(
        f"Churned Customers: "
        f"{churned_customers}"
    )

    print(
        f"Churn Rate: "
        f"{churn_rate:.2f}%"
    )

    print(
        f"Average Monthly Charges: "
        f"{average_monthly_charges:.2f}"
    )

    print(
        f"Highest Churn Segment: "
        f"{highest_churn_segment}"
    )

 
# MAIN PROGRAM 

def main():

    print("=" * 60)
    print("AI/ML CUSTOMER CHURN DATASET PREPARATION SYSTEM")
    print("=" * 60)


    # Create dataset

    create_customer_dataset()


    # Load

    raw_data = load_dataset()


    # Inspect

    inspect_dataset(
        raw_data
    )


    # Clean

    (
        clean_data,
        missing_before,
        missing_after,
        removed_duplicates
    ) = clean_dataset(
        raw_data
    )


    # Filter

    filter_customers(
        clean_data
    )


    # Group

    group_and_analyze(
        clean_data
    )


    # ML Dataset

    X, y = prepare_ml_dataset(
        clean_data
    )


    # Final Report

    final_report(
        raw_data,
        clean_data,
        removed_duplicates,
        missing_before,
        missing_after
    )


    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)

 
# RUN PROGRAM 

if __name__ == "__main__":
    main()