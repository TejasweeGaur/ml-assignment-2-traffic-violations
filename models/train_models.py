import glob
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from data_analysis import analyze_data
from data_preprocessing import build_preprocessor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def train_machine_learning_models():
    """
    Train multiple machine learning models, evaluate their performance, and save the results.
    This function orchestrates the entire machine learning workflow, including data preprocessing,
    model training, evaluation, and artifact saving. It supports multiple models and generates
    performance metrics for comparison.
    Steps:
    1. Create necessary directories for storing artifacts.
    2. Clear any existing artifacts from previous runs.
    3. Analyze the dataset to fetch metadata such as the target column, numerical features,
       and categorical features.
    4. Load training and testing datasets.
    5. Split the datasets into features (X) and target (y).
    6. Encode the target labels if they are categorical.
    7. Save the label encoder for future use.
    8. Build a preprocessing pipeline for numerical and categorical features.
    9. Define a set of machine learning models to train.
    10. For each model:
        - Create a full machine learning pipeline with preprocessing and the model.
        - Train the model on the training data.
        - Evaluate the model on the testing data and collect performance metrics.
        - Save the confusion matrix and classification report for the model.
    11. Compare the performance of all models and save the results to a CSV file.
    12. Display confusion matrices and classification reports for all models.
    Artifacts:
    - Label encoder saved as a pickle file.
    - Trained model pipelines saved for each model.
    - Confusion matrices and classification reports for each model.
    - A CSV file comparing the performance of all models.
    Returns:
        None
    """
    print("*" * 50 + " Starting Machine Learning Models training process " + "*" * 50)

    # Paths
    artifacts_dir, confusion_matrix_dir, classification_report_dir = (
        create_artifacts_directories()
    )

    # Clear existing artifacts
    clear_existing_artifacts()

    # Fetching metadata
    target_column, numerical_features, categorical_features = (
        analyze_and_return_metadata()
    )

    # Load data
    train_df, test_df = load_training_testing_data()

    # Splitting features and target
    X_train, y_train, X_test, y_test = split_features_and_target(
        target_column, train_df, test_df
    )

    print("Encoding target labels if they are categorical...")
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Saving label encoder
    print("Saving label encoder...")
    joblib.dump(label_encoder, artifacts_dir / "label_encoder.pkl")

    # Building preprocessing pipeline
    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor(
        numeric_features=numerical_features,
        categorical_features=categorical_features,
    )
    print("Preprocessing pipeline built successfully.")

    # Defining machine learning models
    machine_learning_models = {
        "Logistic Regression Model": LogisticRegression(max_iter=1000),
        "KNN Model": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree Model": DecisionTreeClassifier(random_state=42),
        "Random Forest Model": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "Naive Bayes Model": GaussianNB(),
        "XGBoost Model": XGBClassifier(
            objective="multi:softprob",
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        ),
    }

    # Initialize a list to store model performance metrics
    model_performance = []

    # Iterating through models, building pipelines, training, and saving them
    for model_name, model_instance in machine_learning_models.items():
        print("*" * 50 + f" Training {model_name} " + "*" * 50)

        # Full ML pipeline
        print("Creating full ML pipeline...")
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("classifier", model_instance),
            ]
        )
        print("Pipeline created successfully.")

        # Train model
        training_time = train_and_save_model(X_train, y_train_enc, model_name, pipeline, artifacts_dir)

        y_pred = evaluate_model_metrics(
            X_test, y_test_enc, model_performance, model_name, pipeline, training_time
        )

        # Saving individual confusion matrix and classification report
        save_confusion_matrix_and_report(
            artifacts_dir, label_encoder, y_test_enc, model_name, y_pred
        )

    results_df = pd.DataFrame(model_performance)
    print("\nModel Comparison:")
    print(results_df)

    show_confusion_and_classification_reports(
        confusion_matrix_dir, classification_report_dir, machine_learning_models
    )

    results_df.to_csv(artifacts_dir / "model_comparison.csv", index=False)


def create_artifacts_directories():
    """
    Creates necessary directories for storing model artifacts if they do not already exist.
    This function ensures that the following directories are created:
    1. `models/artifacts` - The main directory for storing all artifacts related to the model.
    2. `models/artifacts/confusion_matrices` - A subdirectory for storing confusion matrix files.
    3. `models/artifacts/classification_reports` - A subdirectory for storing classification report files.
    If any of these directories already exist, the function does nothing for those directories.
    Returns:
        tuple: A tuple containing three `Path` objects:
            - `artifacts_dir`: Path to the main artifacts directory.
            - `confusion_matrix_dir`: Path to the confusion matrices directory.
            - `classification_report_dir`: Path to the classification reports directory.
    Example:
        artifacts_dir, confusion_matrix_dir, classification_report_dir = create_artifacts_directories()
    """
    print("Creating artifacts directory if it does not exist...")
    artifacts_dir = Path("models/artifacts")
    confusion_matrix_dir = Path("models/artifacts/confusion_matrices")
    classification_report_dir = Path("models/artifacts/classification_reports")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
    classification_report_dir.mkdir(parents=True, exist_ok=True)

    return artifacts_dir, confusion_matrix_dir, classification_report_dir


def show_confusion_and_classification_reports(
    confusion_matrix_dir, classification_report_dir, machine_learning_models
):
    """
    Displays the confusion matrix and classification report for each machine learning model.
    This function reads pre-saved confusion matrices and classification reports for a set of
    machine learning models from the specified directories. It then prints these reports
    to the console for easy viewing and comparison.
    Args:
        confusion_matrix_dir (Path or str): The directory where confusion matrix CSV files
            are stored. Each file should be named in the format
            "confusion_matrix_<model_name>.csv", where <model_name> is the lowercase,
            space-replaced-with-underscore version of the model's name.
        classification_report_dir (Path or str): The directory where classification report
            CSV files are stored. Each file should be named in the format
            "classification_report_<model_name>.csv", where <model_name> is the lowercase,
            space-replaced-with-underscore version of the model's name.
        machine_learning_models (dict): A dictionary where the keys are the names of the
            machine learning models (as strings) and the values are the corresponding model
            objects. The model objects themselves are not used in this function, only the
            names are used to locate the files.
    Prints:
        For each model:
            - The confusion matrix as a DataFrame.
            - The classification report as a DataFrame.
    """

    print("\nDisplaying confusion matrix and classification report for each model:")
    for model_name in machine_learning_models.keys():
        print("\n")
        print("=" * 30 + f" {model_name} " + "=" * 30)

        # Load confusion matrix
        confusion_matrix_path = (
            confusion_matrix_dir
            / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.csv"
        )
        confusion_matrix_df = pd.read_csv(confusion_matrix_path, index_col=0)
        print("Confusion Matrix:")
        print(confusion_matrix_df)

        # Load classification report
        classification_report_path = (
            classification_report_dir
            / f"classification_report_{model_name.lower().replace(' ', '_')}.csv"
        )
        classification_report_df = pd.read_csv(classification_report_path, index_col=0)
        print("\nClassification Report:")
        print(classification_report_df)


def save_confusion_matrix_and_report(
    artifacts_dir, label_encoder, y_test_enc, model_name, y_pred
):
    """
    Saves the confusion matrix and classification report for a given model.
    This function generates a confusion matrix and a classification report
    based on the true labels and predicted labels of a model. It then saves
    these results as CSV files in the specified directory.
    Args:
        artifacts_dir (Path or str): The directory where the confusion matrix
            and classification report will be saved. It should contain subdirectories
            named "confusion_matrices" and "classification_reports".
        label_encoder (LabelEncoder): An encoder that maps label indices to their
            corresponding class names. Used to label rows and columns in the confusion matrix.
        y_test_enc (array-like): The true labels of the test dataset, encoded as integers.
        model_name (str): The name of the model for which the confusion matrix and
            classification report are being generated. This name will be used in the
            filenames of the saved CSV files.
        y_pred (array-like): The predicted labels generated by the model, encoded as integers.
    Returns:
        None: This function does not return any value. It saves the confusion matrix
        and classification report as CSV files in the specified directory.
    Raises:
        ValueError: If the input data is invalid or if the directories for saving
            the files do not exist.
    Side Effects:
        - Saves the confusion matrix as a CSV file in the "confusion_matrices" subdirectory.
        - Saves the classification report as a CSV file in the "classification_reports" subdirectory.
        - Prints messages to indicate the progress and completion of the saving process.
    """

    print(
        "Saving confusion matrix and classification report for the model : "
        + model_name
    )
    confusion_matrix_per_model = confusion_matrix(y_test_enc, y_pred).tolist()
    classification_report_per_model = classification_report(
        y_test_enc,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix_per_model,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )
    classification_report_df = pd.DataFrame(classification_report_per_model).transpose()

    confusion_matrix_df.to_csv(
        artifacts_dir
        / "confusion_matrices"
        / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.csv"
    )
    classification_report_df.to_csv(
        artifacts_dir
        / "classification_reports"
        / f"classification_report_{model_name.lower().replace(' ', '_')}.csv"
    )
    print(f"Confusion matrix and classification report saved for {model_name}.\n")


def evaluate_model_metrics(
    X_test, y_test_enc, model_performance, model_name, pipeline, training_time
):
    """
    Evaluates the performance of a machine learning model on test data and appends the metrics to a list.
    Parameters:
    -----------
    X_test : array-like
        The test dataset containing the features used for prediction.
    y_test_enc : array-like
        The true labels for the test dataset, encoded as integers or one-hot vectors.
    model_performance : list
        A list to store the performance metrics of the evaluated model.
    model_name : str
        The name of the model being evaluated, used for identification in the metrics.
    pipeline : sklearn.pipeline.Pipeline
        The trained machine learning pipeline or model used for making predictions.
    Returns:
    --------
    y_pred : array-like
        The predicted labels for the test dataset.
    Side Effects:
    -------------
    Appends a dictionary containing the following performance metrics to the `model_performance` list:
        - "Model": The name of the model.
        - "Accuracy": The proportion of correct predictions out of all predictions.
        - "Precision": The average precision score across all classes (macro-averaged).
        - "Recall": The average recall score across all classes (macro-averaged).
        - "F1": The average F1 score across all classes (macro-averaged).
        - "AUC": The Area Under the Curve (AUC) score for multi-class classification.
        - "MCC": The Matthews Correlation Coefficient, a balanced measure of model quality.
    This function calculates and stores these metrics to help evaluate and compare the performance of different models.
    """

    print("Calculating model performance on test data...")
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    end_time = time.time()
    evaluation_time = end_time - start_time

    model_metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test_enc, y_pred),
        "Precision": precision_score(y_test_enc, y_pred, average="macro"),
        "Recall": recall_score(y_test_enc, y_pred, average="macro"),
        "F1": f1_score(y_test_enc, y_pred, average="macro"),
        "AUC": roc_auc_score(y_test_enc, y_prob, multi_class="ovr"),
        "MCC": matthews_corrcoef(y_test_enc, y_pred),
        "Training Time (s)": training_time,
        "Evaluation Time (s)": evaluation_time,
    }
    model_performance.append(model_metrics)
    return y_pred


def train_and_save_model(X_train, y_train_enc, model_name, pipeline, artifacts_dir):
    """
    Trains a machine learning model using the provided training data and pipeline,
    and saves the trained model to a file.
    Args:
        X_train (array-like): The input features for training the model.
            This should be a 2D array where each row represents a training sample
            and each column represents a feature.
        y_train_enc (array-like): The target labels for training the model.
            This should be a 1D array where each element corresponds to the label
            of a training sample.
        model_name (str): The name of the model being trained. This will be used
            to identify the model and to name the saved file.
        pipeline (Pipeline): A machine learning pipeline that includes preprocessing
            steps and the model to be trained. The pipeline should be compatible
            with the `fit` method.
        artifacts_dir (Path): The directory where the trained model will be saved.
    Behavior:
        - Prints messages to indicate the progress of training and saving the model.
        - Trains the model using the provided training data and pipeline.
        - Saves the trained model to a file in the specified artifacts directory.
          The filename is generated based on the model name, converted to lowercase
          and spaces replaced with underscores, with a ".pkl" extension.
    Raises:
        Exception: If there is an error while saving the trained model to a file,
            an error message is printed, and the function returns without saving.
    """

    print(f"Training {model_name}...")
    start_time = time.time()
    pipeline.fit(X_train, np.array(y_train_enc))
    end_time = time.time()
    training_time = end_time - start_time
    print(f"{model_name} trained in {training_time:.2f} seconds.")
    print("Model training completed.")

    # Save trained model
    try:
        print(f"Saving trained {model_name} to file...")
        model_filename = artifacts_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(pipeline, model_filename)
        print(f"Trained model saved as {model_filename}.\n")
        return training_time
    except Exception as e:
        print(f"Error saving model {model_name}: {e}")
        return


def split_features_and_target(target_column, train_df, test_df):
    """
    Splits the input training and testing data into features and target variables.
    This function takes a target column name and two dataframes (training and testing),
    and separates the target column from the rest of the columns in both dataframes.
    The target column is treated as the variable to be predicted, while the remaining
    columns are treated as the features used for prediction.
    Args:
        target_column (str): The name of the column to be used as the target variable.
        train_df (pd.DataFrame): The dataframe containing the training data.
        test_df (pd.DataFrame): The dataframe containing the testing data.
    Returns:
        tuple: A tuple containing four elements:
            - X_train (pd.DataFrame): The training features (all columns except the target column).
            - y_train (pd.Series): The training target (the target column from the training dataframe).
            - X_test (pd.DataFrame): The testing features (all columns except the target column).
            - y_test (pd.Series): The testing target (the target column from the testing dataframe).
    Prints:
        - The shapes of the training features, training target, testing features, and testing target
          to provide a quick overview of the data split.
    """

    print("Splitting features and target...")
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Testing target shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test


def load_training_testing_data():
    """
    Loads the training and testing datasets from CSV files and prints their shapes.
    This function reads two CSV files, 'train.csv' and 'test.csv', located in the
    'datasets' directory. It loads the data into pandas DataFrames and displays
    the number of rows and columns in each dataset. The loaded DataFrames are
    then returned for further processing.
    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - train_df: The DataFrame containing the training dataset.
            - test_df: The DataFrame containing the testing dataset.
    """

    print("Loading training and testing datasets...")
    train_df = pd.read_csv("datasets/train.csv")
    test_df = pd.read_csv("datasets/test.csv")
    print(f"Training dataset shape: {train_df.shape}")
    print(f"Testing dataset shape: {test_df.shape}")
    return train_df, test_df


def analyze_and_return_metadata():
    """
    Analyzes data to extract and return metadata information.
    This function performs an analysis on the dataset to identify and retrieve
    key metadata, including the target column, numerical features, and categorical features.
    It also prints the extracted metadata for reference.
    Returns:
        tuple: A tuple containing the following elements:
            - target_column (str): The name of the target column in the dataset.
            - numerical_features (list): A list of numerical feature names in the dataset.
            - categorical_features (list): A list of categorical feature names in the dataset.
    """

    print("Analyzing data to fetch metadata...")
    (_, target_column, _, numerical_features, categorical_features, _) = analyze_data(
        display_comments=False
    )
    print(f"Target column: {target_column}")
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    return target_column, numerical_features, categorical_features


def clear_existing_artifacts():
    """
    Clears existing model artifacts from the specified directories.
    This function is responsible for cleaning up old or unnecessary model artifacts
    such as pickle files (*.pkl) and CSV files (*.csv) from the designated directories.
    It ensures that the workspace is clean and ready for new model training or evaluation.
    The function performs the following steps:
    1. Defines the base directory where model artifacts are stored.
    2. Specifies subdirectories and file patterns to look for.
    3. Iterates through each subdirectory and file pattern to locate matching files.
    4. Deletes the identified files, if any, and logs the deletion process.
    5. If no files are found, it logs a message indicating the absence of files.
    Key components:
    - `base_artifacts_dir`: The root directory where model artifacts are stored.
    - `sub_dirs`: A list of subdirectories within the base directory to search for files.
    - `file_patterns`: A list of file patterns (e.g., *.pkl, *.csv) to identify files for deletion.
    Logs are printed to indicate the progress of the cleanup process, including:
    - Messages for each subdirectory and file pattern being processed.
    - Notifications for files that are successfully deleted.
    - Warnings for any errors encountered during file deletion.
    - Messages indicating when no files are found in a specific directory.
    This function is useful for maintaining a clean and organized workspace by removing outdated or redundant files.
    """

    print("*" * 50 + " Clearing existing model artifacts " + "*" * 50)
    base_artifacts_dir = "models/artifacts"
    sub_dirs = ["", "confusion_matrices", "classification_reports"]
    file_patterns = ["*.pkl", "*.csv"]

    for sub_dir in sub_dirs:
        artifacts_dir = os.path.join(base_artifacts_dir, sub_dir)
        for pattern in file_patterns:
            files = glob.glob(os.path.join(artifacts_dir, pattern))
            if not files:
                print(f"\tNo existing {pattern} files found in '{artifacts_dir}'.")
                continue

            for file_path in files:
                try:
                    os.remove(file_path)
                    print(f"\tDeleted: {file_path}")
                except Exception as e:
                    print(f"\tError deleting {file_path}: {e}")

    print("*" * 50 + " Existing model artifacts cleared " + "*" * 50)


if __name__ == "__main__":
    train_machine_learning_models()
