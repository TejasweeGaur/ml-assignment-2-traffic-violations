"""
Setup and manage dataset downloads from Kaggle for the traffic violations analysis.
This module provides utilities to download the Indian Traffic Violation dataset from Kaggle
using the kagglehub library. It includes functions to ensure directories exist, check if
datasets are already downloaded, and download datasets while organizing files appropriately.
Module-level constants:
    DOWNLOAD_DATASET_TO (str): Target directory for dataset downloads ("datasets")
    DATASET_TO_DOWNLOAD (str): Kaggle dataset identifier ("khushikyad001/indian-traffic-violation")
Functions:
    ensure_directory_exists(directory_path): Creates a directory if it doesn't exist
    is_dataset_downloaded(dataset_path): Checks if dataset exists and is not empty
    download_dataset(): Downloads and organizes the dataset from Kaggle
Usage:
    Run this script directly to automatically download the dataset if not already present:
    $ python setup_dataset.py
Dependencies:
    - kagglehub: For downloading datasets from Kaggle
    - os: For file system operations
Example:
    The script will automatically:
    1. Create the datasets directory if needed
    2. Check if the dataset already exists
    3. Download from Kaggle if not present
    4. Organize downloaded files in the target directory
"""

import glob
import os
import shutil

import kagglehub

DOWNLOAD_DATASET_TO = "datasets"
DATASET_TO_DOWNLOAD = "khushikyad001/indian-traffic-violation"


def clear_kaggle_cache():
    """Clears the Kaggle cache to force a fresh download of the dataset."""
    kaggle_cache_dir = os.path.expanduser("~/.kaggle/datasets")
    kagglehub_cache_dir = os.path.expanduser("~/.cache/kagglehub/datasets")
    if os.path.exists(kaggle_cache_dir):
        print(f"Clearing Kaggle cache at {kaggle_cache_dir}...")
        shutil.rmtree(kaggle_cache_dir)
        print("Kaggle cache cleared.")
    else:
        print("No Kaggle cache found to clear.")

    if os.path.exists(kagglehub_cache_dir):
        print(f"Clearing KaggleHub cache at {kagglehub_cache_dir}...")
        shutil.rmtree(kagglehub_cache_dir)
        print("KaggleHub cache cleared.")
    else:
        print("No KaggleHub cache found to clear.")


def ensure_directory_exists(directory_path):
    """Checks if the directory to download the Dataset to exists. If not then create a directory

    Args:
        directory_path (str): Path to the directory to check/create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def is_dataset_downloaded(dataset_path):
    """Checks if the dataset has been downloaded already or not.

    Args:
        dataset_path (str): Path to the dataset directory

    Returns:
        bool: True if the dataset directory exists and is not empty, False otherwise
    """
    dataset_file = os.path.join(dataset_path, "dataset.csv")
    return os.path.exists(dataset_path) and os.path.isfile(dataset_file)


def download_dataset():
    """
    Download a dataset from Kaggle and organize it in the target directory.
    This function downloads a dataset specified by DATASET_TO_DOWNLOAD from Kaggle
    using the kagglehub library and saves it to DOWNLOAD_DATASET_TO. After download,
    it moves all files from the dataset's subdirectory to the main download directory.
    The function performs the following steps:
    1. Downloads the dataset from Kaggle to a temporary location
    2. Iterates through all files in the downloaded directory
    3. Moves each file to the target download directory if it doesn't already exist
    4. Prints status messages at each step
    Raises:
        Exception: If the Kaggle API is not properly configured or if the dataset
                   cannot be found on Kaggle.
    Note:
        - Requires DATASET_TO_DOWNLOAD and DOWNLOAD_DATASET_TO to be defined globally
        - Requires kagglehub and os modules to be imported
        - Files that already exist in the target directory are not overwritten
    """
    print(
        f"Downloading dataset {DATASET_TO_DOWNLOAD} from Kaggle to {DOWNLOAD_DATASET_TO}/..."
    )

    # Download into the target directory
    path = kagglehub.dataset_download(DATASET_TO_DOWNLOAD)
    print(f"Dataset downloaded to {path}")

    dst_csv = os.path.join(DOWNLOAD_DATASET_TO, "dataset.csv")
    if os.path.isfile(dst_csv):
        print(f"{dst_csv} already exists, skipping move.")
        return

    # If the returned path is a file (zip or csv), handle accordingly
    if os.path.isfile(path):
        lower = path.lower()
        if lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            print(f"Extracting archive {path}...")
            try:
                shutil.unpack_archive(path, DOWNLOAD_DATASET_TO)
            except shutil.ReadError as e:
                print(
                    f"Archive extraction failed ({e}), continuing to scan for CSV files."
                )
        elif lower.endswith(".csv"):
            print(f"Moving CSV {path} to {dst_csv}...")
            shutil.move(path, dst_csv)
            print("Dataset downloaded and moved successfully.")
            return

    # If path is a directory (or after extraction), search for CSV files
    search_dir = path if os.path.isdir(path) else DOWNLOAD_DATASET_TO
    if os.path.isdir(search_dir):
        archive_files = []
        for ext in ("*.zip", "*.tar", "*.tar.gz", "*.tgz"):
            archive_files.extend(
                glob.glob(os.path.join(search_dir, "**", ext), recursive=True)
            )
        for archive_path in archive_files:
            print(f"Extracting archive {archive_path}...")
            try:
                shutil.unpack_archive(archive_path, DOWNLOAD_DATASET_TO)
            except shutil.ReadError as e:
                print(f"Archive extraction failed ({e}), continuing.")
    print(f"Searching for CSV files in {search_dir}...")
    print(
        f"Contents of {search_dir}: {os.listdir(search_dir) if os.path.exists(search_dir) else 'Directory does not exist'}"
    )
    csv_files = glob.glob(os.path.join(search_dir, "**", "*.csv"), recursive=True)
    if os.path.isdir(DOWNLOAD_DATASET_TO):
        csv_files.extend(
            glob.glob(os.path.join(DOWNLOAD_DATASET_TO, "**", "*.csv"), recursive=True)
        )
    if not csv_files:
        # Check if this was an archive file that may have failed extraction
        extraction_note = ""
        if os.path.isfile(path):
            lower = path.lower()
            if lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                extraction_note = " (archive extraction may have failed)"
        raise Exception(
            "No CSV file found in the downloaded dataset"
            f"{extraction_note}. "
            "Ensure your Kaggle credentials are configured, you have accepted the "
            "dataset rules on Kaggle, and the download completed successfully."
        )

    # Prefer a single dataset CSV; take the first match
    src_csv = csv_files[0]
    if os.path.abspath(src_csv) != os.path.abspath(dst_csv):
        print(f"Moving {src_csv} to {dst_csv}...")
        shutil.move(src_csv, dst_csv)

    print("Dataset downloaded and moved successfully.")


if __name__ == "__main__":
    ensure_directory_exists(DOWNLOAD_DATASET_TO)
    clear_kaggle_cache()
    if not is_dataset_downloaded(DOWNLOAD_DATASET_TO):
        print(f"Dataset not found in {DOWNLOAD_DATASET_TO}/, downloading now...")
        download_dataset()
    else:
        print(f"Dataset already exists in {DOWNLOAD_DATASET_TO}, skipping download.")

    print("Dataset setup complete.")
