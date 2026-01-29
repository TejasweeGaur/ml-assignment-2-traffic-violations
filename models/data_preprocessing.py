import data_analysis as da
import pandas as pd


def preprocess_data():
    # Analyze data to get necessary information
    (
        df,
        raw_feature_columns,
        target_column,
        columns_to_drop,
        features_to_encode,
        features_to_impute,
        numerical_features,
        date_and_time_features,
    ) = da.analyze_data()

    print("\n" + "-" * 50 + " Starting Data Preprocessing " + "-" * 50)
    # Extracting Date and Time Features
    print("\nExtracting Date and Time Features...")
    df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df["hour_of_day"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour
    print("Date and Time Features extracted: 'day_of_week', 'hour_of_day'")
    print(df.head()[["Date", "Time", "day_of_week", "hour_of_day"]].head())
    print("Date and Time Features extraction complete.")

    print("\n" + "-" * 50 + " Data Preprocessing Complete " + "-" * 50)

    return df


if __name__ == "__main__":
    preprocessed_df = preprocess_data()
