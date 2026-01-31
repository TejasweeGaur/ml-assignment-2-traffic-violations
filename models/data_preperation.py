import data_analysis as da
import pandas as pd
from sklearn.model_selection import train_test_split


def data_preparation():
    """
    Prepares the data for machine learning by performing the following steps:
    1. Fetches data and metadata from the `data_analysis` module.
    2. Creates a copy of the original dataframe to avoid modifying it directly.
    3. Extracts date and time features (e.g., day of the week, hour of the day) from specified columns.
    4. Drops the original date and time columns after extraction.
    5. Removes unnecessary features specified in the `features_to_drop` list.
    6. Separates the target column from the feature columns.
    7. Splits the data into training and testing sets with an 80-20 split, ensuring stratified sampling.
    8. Saves the training and testing data to CSV files for future use.
    Returns:
        tuple: A tuple containing the following:
            - X_train (pd.DataFrame): The training feature set.
            - X_test (pd.DataFrame): The testing feature set.
            - y_train (pd.Series): The training target values.
            - y_test (pd.Series): The testing target values.
    Raises:
        Any exceptions raised during data fetching, processing, or file operations.
    Dependencies:
        - Requires the `data_analysis` module to provide the data and metadata.
        - Uses pandas for data manipulation and sklearn for data splitting.
    Side Effects:
        - Saves the prepared training data to 'datasets/train.csv'.
        - Saves the prepared testing data to 'datasets/test.csv'.
    """

    # Fetching data from data_analysis module
    print("Fetching data from data_analysis module...")
    (
        df,
        target_column,
        features_to_drop,
        _,
        _,
        date_and_time_features,
    ) = da.analyze_data()
    print("Data fetched successfully.")
    print(f"Target column: {target_column}")
    print(f"Features to drop: {features_to_drop}")
    print(f"Date and time features: {date_and_time_features}")

    print("*" * 50 + " Starting data preparation " + "*" * 50)
    # Assigning df to new variable so as to avoid modifying original dataframe
    print("Creating a copy of the original dataframe...")
    df_new = df.copy()

    # Extracting date and time features
    print("Extracting date and time features...")
    for data_time_feature in date_and_time_features:
        if data_time_feature == "Date":
            print(f"Processing date feature: {data_time_feature}")
            df_new["day_of_week"] = pd.to_datetime(
                df[data_time_feature], errors="coerce"
            ).dt.dayofweek
        elif data_time_feature == "Time":
            print(f"Processing time feature: {data_time_feature}")
            df_new["hour"] = pd.to_datetime(
                df[data_time_feature], errors="coerce"
            ).dt.hour

    # Dropping original date and time features
    print("Dropping original date and time features...")
    df_new = df_new.drop(columns=date_and_time_features)

    # Dropping specified features
    print("Dropping specified features...")
    df_new = df_new.drop(columns=features_to_drop)

    # Dropping Target Column
    print("Separating target column from features...")
    X = df_new.drop(target_column, axis=1)
    y = df_new[target_column]

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print("Data split completed.")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Saving the prepared data to CSV files
    print("Saving training and testing data to CSV files...")
    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    train_df.to_csv("datasets/train.csv", index=False)
    print("Training data saved to 'datasets/train.csv'")
    test_df.to_csv("datasets/test.csv", index=False)
    print("Testing data saved to 'datasets/test.csv'")

    print("*" * 50 + " Data preparation completed " + "*" * 50)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_preparation()
