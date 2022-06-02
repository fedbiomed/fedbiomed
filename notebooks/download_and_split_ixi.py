#!/usr/bin/env python
import hashlib
import os
import requests
from tqdm import tqdm
import argparse
from zipfile import ZipFile
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description='IXI Sample downloader and splitter')
    parser.add_argument('-f', '--root_folder', required=True, type=str)

    return parser.parse_args()


def has_correct_checksum_md5(filename, hash):
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return str(file_hash) == hash


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    print('Downloading file from:', url)
    print('File will be saved as:', filename)
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


def download_and_extract_ixi_sample(root_folder):
    url = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/7kd5wj7v7p-3.zip'
    zip_filename = os.path.join(root_folder, '7kd5wj7v7p-3.zip')
    data_folder = os.path.join(root_folder, '7kd5wj7v7p-3', 'IXI_sample')

    # Check if extracted folder exists
    if os.path.isdir(data_folder):
        print(f'Dataset folder already exists in {data_folder}')
        return data_folder

    # Extract if ZIP exists but not folder
    if not os.path.exists(zip_filename) and has_correct_checksum_md5(zip_filename, 'eecb83422a2685937a955251fa45cb03'):
        # Download if it does not exist
        download_file(url, zip_filename)

    if not os.path.isdir(data_folder):
        with ZipFile(zip_filename, 'r') as zip_obj:
            zip_obj.extractall(root_folder)

    assert os.path.isdir(data_folder)
    return data_folder


if __name__ == '__main__':
    args = parse_args()
    root_folder = os.path.abspath(os.path.expanduser(args.root_folder))
    assert os.path.isdir(root_folder), f'Folder does not exist: {root_folder}'

    # Centralized dataset
    centralized_data_folder = download_and_extract_ixi_sample(root_folder)

    # Federated Dataset
    dst_folder_base = os.path.join(root_folder, 'UniCancer-Centers')

    csv_global = os.path.join(centralized_data_folder, 'participants.csv')
    allcenters = pd.read_csv(csv_global)

    # Split centers
    center_names = ['Guys', 'HH', 'IOP']
    center_dfs = list()

    for center_name in center_names:
        df = allcenters[allcenters.SITE_NAME == center_name]
        center_dfs.append(df)

        train, test = train_test_split(df, test_size=0.1)

        train_folder = os.path.join(dst_folder_base, center_name, 'train')
        holdout_folder = os.path.join(dst_folder_base, center_name, 'holdout')
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(holdout_folder):
            os.makedirs(holdout_folder)

        for subject_folder in train.FOLDER_NAME.values:
            shutil.copytree(
                src=os.path.join(centralized_data_folder, subject_folder),
                dst=os.path.join(train_folder, subject_folder),
                dirs_exist_ok=True
            )

        train.to_csv(os.path.join(train_folder, 'participants.csv'))

        for subject_folder in test.FOLDER_NAME.values:
            shutil.copytree(
                src=os.path.join(centralized_data_folder, subject_folder),
                dst=os.path.join(holdout_folder, subject_folder),
                dirs_exist_ok=True
            )
        test.to_csv(os.path.join(holdout_folder, 'participants.csv'))
