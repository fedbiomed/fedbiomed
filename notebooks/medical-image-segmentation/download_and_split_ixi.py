#!/usr/bin/env python
import argparse
import hashlib
import os
import shutil
from zipfile import ZipFile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fedbiomed.node.config import node_component


def parse_args():
    parser = argparse.ArgumentParser(description="IXI Sample downloader and splitter")
    parser.add_argument("-f", "--root_folder", required=True, type=str)
    parser.add_argument(
        "-F",
        "--force",
        action=argparse.BooleanOptionalAction,
        required=False,
        type=bool,
        default=False,
    )
    return parser.parse_args()


def has_correct_checksum_md5(filename, hash):
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return str(file_hash.hexdigest()) == hash


def download_file(url, filename):
    """
    Download a large file from `url` and save it to `filename`.
    Raises requests.HTTPError if the server returns an error status.
    """
    print(f"Downloading file from: {url}")
    print(f"File will be saved as: {filename}")
    chunk_size = 1024
    r = requests.get(url, stream=True)
    r.raise_for_status()

    content_length = r.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if total is None:
        print(
            "Warning: server did not provide Content-Length; progress bar will be indeterminate."
        )

    with open(filename, "wb") as f:
        pbar = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


def download_and_extract_ixi_sample(root_folder):
    url = "https://data.mendeley.com/public-api/zip/7kd5wj7v7p/download/3"
    zip_filename = os.path.join(root_folder, "notebooks", "data", "7kd5wj7v7p-3.zip")
    data_folder = os.path.join(root_folder, "notebooks", "data")
    extracted_folder = os.path.join(data_folder, "7kd5wj7v7p-3", "IXI_sample")

    if not os.path.exists(zip_filename):
        download_file(url, zip_filename)

    # Skip extraction if already done
    if os.path.isdir(extracted_folder):
        print(f"Dataset folder already exists at: {extracted_folder}")
        return extracted_folder

    print("Verifying checksum of downloaded ZIP...")
    expected_md5 = "eecb83422a2685937a955251fa45cb03"
    if not has_correct_checksum_md5(zip_filename, expected_md5):
        raise ValueError(
            f"MD5 checksum mismatch for '{zip_filename}'. "
            "The file may be corrupted or incomplete. "
            "Delete it and re-run the script to download it again."
        )
    print("Checksum OK. Extracting ZIP...")

    with ZipFile(zip_filename, "r") as zip_obj:
        zip_obj.extractall(data_folder)

    if not os.path.isdir(extracted_folder):
        raise RuntimeError(
            f"Extraction completed but expected folder was not found: '{extracted_folder}'. "
            "The ZIP archive may have a different internal structure than expected."
        )

    print(f"Extraction complete. Dataset located at: {extracted_folder}")
    return extracted_folder


if __name__ == "__main__":
    args = parse_args()
    root_folder = os.path.abspath(os.path.expanduser(args.root_folder))

    if not os.path.isdir(root_folder):
        raise SystemExit(f"Error: root folder does not exist: '{root_folder}'")

    # Download and extract the centralized dataset
    print("--- Step 1: Downloading and extracting IXI sample dataset ---")
    centralized_data_folder = download_and_extract_ixi_sample(root_folder)

    # Prepare the federated split output folder
    federated_data_folder = os.path.join(
        root_folder, "notebooks", "data", "Hospital-Centers"
    )
    shutil.rmtree(federated_data_folder, ignore_errors=True)

    csv_global = os.path.join(centralized_data_folder, "participants.csv")
    if not os.path.isfile(csv_global):
        raise FileNotFoundError(
            f"Expected participants CSV not found: '{csv_global}'. "
            "The dataset may not have been extracted correctly."
        )
    allcenters = pd.read_csv(csv_global)

    # Split into per-center federated nodes
    print("\n--- Step 2: Splitting dataset into federated center nodes ---")
    center_names = ["Guys", "HH", "IOP"]
    center_dfs = list()

    for center_name in center_names:
        component_folder = os.path.join(os.getcwd(), center_name.lower())

        print(f"\nProcessing center '{center_name}' -> {component_folder}")
        if node_component.is_component_existing(component_folder) and not args.force:
            print(
                f"Warning: component '{component_folder}' already exists. "
                "Skipping node initialisation. Use --force to overwrite."
            )
        else:
            node_component.initiate(component_folder)

        df = allcenters[allcenters.SITE_NAME == center_name]
        if df.empty:
            raise ValueError(
                f"No subjects found for center '{center_name}' in '{csv_global}'. "
                "Check that SITE_NAME values match the expected center names."
            )
        center_dfs.append(df)

        train, test = train_test_split(df, test_size=0.1, random_state=21)
        print(f"  Train subjects: {len(train)}, holdout subjects: {len(test)}")

        train_folder = os.path.join(os.getcwd(), center_name.lower(), "data", "train")
        holdout_folder = os.path.join(
            os.getcwd(), center_name.lower(), "data", "holdout"
        )

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(holdout_folder, exist_ok=True)

        for subject_folder in train.FOLDER_NAME.values:
            shutil.copytree(
                src=os.path.join(centralized_data_folder, subject_folder),
                dst=os.path.join(train_folder, subject_folder),
                dirs_exist_ok=True,
            )
        train.to_csv(os.path.join(train_folder, "participants.csv"))

        for subject_folder in test.FOLDER_NAME.values:
            shutil.copytree(
                src=os.path.join(centralized_data_folder, subject_folder),
                dst=os.path.join(holdout_folder, subject_folder),
                dirs_exist_ok=True,
            )
        test.to_csv(os.path.join(holdout_folder, "participants.csv"))

    print("\n--- Setup complete ---")
    print(f"Centralized dataset: {centralized_data_folder}")
    print(f"Federated dataset:   {federated_data_folder}")

    print(
        "\nTo add data to each node (select 'medical folder dataset' and use tag 'ixi-train'):"
    )
    for center_name in center_names:
        print(f"\tfedbiomed node -p {center_name.lower()} dataset add")

    print("\nTo start each node:")
    for center_name in center_names:
        print(f"\tfedbiomed node -p {center_name.lower()} start")
