import pandas as pd


def analyze_data():
    data_path = "datasets/dataset.csv"
    df = pd.read_csv(data_path)

    print("-" * 50, "Data Summary:", "-" * 50)
    print("\nDescribe Dataset => \n", df.describe())
    print("\nDataset Shape => \n", df.shape)
    print("\nDataset Columns/Features => \n", df.columns)

    print("\nMissing Values => \n", df.isnull().sum())

    print("\nUnique Values per Column => \n", df.nunique())

    print("\nUnique Violations => \n", df["Violation_Type"].value_counts())
    print("\nUnique Locations => \n", df["Location"].value_counts())
    print("\nUnique Vehicle Types => \n", df["Vehicle_Type"].value_counts())
    print("\nUnique Vehicle Colors => \n", df["Vehicle_Color"].value_counts())
    print("\nUnique Registration States => \n", df["Registration_State"].value_counts())
    print("\nUnique Driver Gender => \n", df["Driver_Gender"].value_counts())
    print("\nUnique License Type => \n", df["License_Type"].value_counts())
    print("\nUnique Weather Condition => \n", df["Weather_Condition"].value_counts())
    print("\nUnique Road Condition => \n", df["Road_Condition"].value_counts())
    print(
        "\nUnique Traffic Light Status => \n", df["Traffic_Light_Status"].value_counts()
    )

    # Data Analysis based on above insights - type of features and their potential use cases
    # "Violation_ID",  # Unique identifier for each violation - can be dropped
    # "Violation_Type",  # Categorical feature indicating type of violation - target variable
    # "Fine_Amount",  # Numerical feature indicating fine amount - can be dropped since it tells about severity of violation
    # "Location",  # Categorical feature indicating location of violation - can be used for geospatial analysis - can be encoded into state codes
    # "Date",  # Date feature - can be used to extract day, month, year for analyzing the data around weekends, holidays etc. leading to violations
    # "Time",  # Time feature - can be used to extract hour, minute for analyzing the data around early mornings, late nights etc. leading to violations
    # "Vehicle_Type",  # Categorical Feature; will be handled by one-hot encoding in preprocessing pipeline
    # "Vehicle_Color",  # Categorical feature indicating color of vehicle - can be used for classification analysis - can be encoded into color categories
    # "Vehicle_Model_Year",  # Numerical feature indicating model year of vehicle - can be used for numerical feature capturing
    # "Registration_State",  # Categorical feature indicating state of registration - can be used for geospatial analysis - can be encoded into state codes
    # "Driver_Age",  # Numerical feature indicating age of driver - can be used for numerical feature capturing
    # "Driver_Gender",  # Categorical feature indicating gender of driver - will be handled by one-hot encoding in preprocessing pipeline
    # "License_Type",  # Categorical feature indicating type of license - will be handled by one-hot encoding in preprocessing pipeline
    # "Penalty_Points",  # Numerical feature indicating penalty points - can be used for numerical feature capturing
    # "Weather_Condition",  # Categorical feature indicating weather condition - will be handled by one-hot encoding in preprocessing pipeline
    # "Road_Condition",  # Categorical feature indicating road condition - will be handled by one-hot encoding in preprocessing pipeline
    # "Officer_ID",  # Categorical feature indicating officer ID - can be dropped as it may not add significant value
    # "Issuing_Agency",  # Categorical feature indicating issuing agency - can be dropped as it may not add significant value
    # "License_Validity",  # Categorical feature indicating license validity - will be handled by one-hot encoding in preprocessing pipeline
    # "Number_of_Passengers",  # Numerical feature indicating number of passengers - can be used for numerical feature capturing
    # "Helmet_Worn",  # Categorical feature indicating if helmet was worn - contains missing values so need to handle accordingly
    # "Seatbelt_Worn",  # Categorical feature indicating if seatbelt was worn - contains missing values so need to handle accordingly
    # "Traffic_Light_Status",  # Categorical feature indicating traffic light status - will be handled by one-hot encoding in preprocessing pipeline
    # "Speed_Limit",  # Numerical feature indicating speed limit - can be used for numerical feature capturing
    # "Recorded_Speed",  # Numerical feature indicating recorded speed - can be used for numerical feature capturing
    # "Alcohol_Level",  # Numerical feature indicating alcohol level - contains missing values so need to handle accordingly
    # "Breathalyzer_Result",  # Categorical feature indicating breathalyzer result - contains missing values so need to handle accordingly
    # "Towed",  # Categorical feature indicating if vehicle was towed - can be used for classification analysis
    # "Fine_Paid",  # Categorical feature indicating if fine was paid - can be dropped as it happens after the violation
    # "Payment_Method",  # Categorical feature indicating payment method - can be dropped as it happens after the violation
    # "Court_Appearance_Required",  # Categorical feature indicating if court appearance is required - can be dropped as it happens after the violation
    # "Previous_Violations",  # Numerical feature indicating number of previous violations - can be used for numerical feature capturing
    # "Comments",  # Text feature containing comments - can be dropped as it may not add significant value

    # Dropping columns that may not add significant value or contain too many missing values
    features_to_drop = [
        "Violation_ID",  # Unique identifier for each violation - can be dropped
        "Fine_Amount",  # Numerical feature indicating fine amount - can be dropped since it tells about severity of violation
        "Officer_ID",  # Categorical feature indicating officer ID - can be dropped as it may not add significant value
        "Issuing_Agency",  # Categorical feature indicating issuing agency - can be dropped as it may not add significant value
        "Fine_Paid",  # Categorical feature indicating if fine was paid - can be dropped as it happens after the violation
        "Payment_Method",  # Categorical feature indicating payment method - can be dropped as it happens after the violation
        "Court_Appearance_Required",  # Categorical feature indicating if court appearance is required - can be dropped as it happens after the violation
        "Comments",  # Text feature containing comments - can be dropped as it may not add significant value
    ]

    # Target Column
    target_column = "Violation_Type"

    # Categorical features
    categorical_features = [
        "Location",  ## Categorical feature indicating location of violation - can be used for geospatial analysis - can be encoded into state codes
        "Vehicle_Type",  # Categorical feature indicating type of vehicle - will be handled by one-hot encoding in preprocessing pipeline
        "Vehicle_Color",  # Categorical feature indicating color of vehicle - will be handled by one-hot encoding in preprocessing pipeline
        "Registration_State",  # Categorical feature indicating state of registration - can be used for geospatial analysis - will be handled by one-hot encoding in preprocessing pipeline
        "Driver_Gender",  # Categorical feature indicating gender of driver - will be handled by one-hot encoding in preprocessing pipeline
        "License_Type",  # Categorical feature indicating type of license - will be handled by one-hot encoding in preprocessing pipeline
        "Weather_Condition",  # Categorical feature indicating weather condition - will be handled by one-hot encoding in preprocessing pipeline
        "Road_Condition",  # Categorical feature indicating road condition - will be handled by one-hot encoding in preprocessing pipeline
        "Traffic_Light_Status",  # Categorical feature indicating traffic light status - will be handled by one-hot encoding in preprocessing pipeline
        "License_Validity",  # Categorical feature indicating license validity - can be used for classification analysis
        "Towed",  # Categorical feature indicating if vehicle was towed - can be used for classification analysis
        "Breathalyzer_Result",  # Categorical feature indicating breathalyzer result - contains missing values so need to handle accordingly
        "Helmet_Worn",  # Categorical feature indicating if helmet was worn - contains missing values so need to handle accordingly
        "Seatbelt_Worn",  # Categorical feature indicating if seatbelt was worn - contains missing values so need to handle accordingly
    ]

    # Numerical features
    numerical_features = [
        "Driver_Age",  # Numerical feature indicating age of driver - can be used for numerical feature capturing
        "Vehicle_Model_Year",  # Numerical feature indicating model year of vehicle - can be used for numerical feature capturing
        "Recorded_Speed",  # Numerical feature indicating recorded speed - can be used for numerical feature capturing
        "Speed_Limit",  # Numerical feature indicating speed limit - can be used for numerical feature capturing
        "Previous_Violations",  # Numerical feature indicating number of previous violations - can be used for numerical feature capturing
        "Penalty_Points",  # Numerical feature indicating penalty points - can be used for numerical feature capturing
        "Number_of_Passengers",  # Numerical feature indicating number of passengers - can be used for numerical feature capturing
        "Alcohol_Level",  # Numerical feature indicating alcohol level - contains missing values so need to handle accordingly
    ]

    # Date and Time features
    date_and_time_features = [
        "Date",  # Date feature - can be used to extract day, month, year for analyzing the data around weekends, holidays etc. leading to violations
        "Time",  # Time feature - can be used to extract hour, minute for analyzing the data around early mornings, late nights etc. leading to violations
    ]

    print("\nTarget Column => \n", target_column)
    print("\nDropping Features => \n", features_to_drop)
    print("\nNumerical Features => \n", numerical_features)
    print("\nCategorical Features => \n", categorical_features)
    print("\nDate and Time Features => \n", date_and_time_features)
    print("\n" + "-" * 50 + " Data Analysis Complete " + "-" * 50)
    return (
        df,
        target_column,
        features_to_drop,
        numerical_features,
        categorical_features,
        date_and_time_features,
    )


if __name__ == "__main__":
    analyze_data()
