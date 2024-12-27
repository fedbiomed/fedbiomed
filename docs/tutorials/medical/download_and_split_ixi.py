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

from fedbiomed.node.config import node_component


def parse_args():
    parser = argparse.ArgumentParser(description='IXI Sample downloader and splitter')
    parser.add_argument('-f', '--root_folder', required=True, type=str)
    parser.add_argument('-F', '--force',  action=argparse.BooleanOptionalAction, required=False, type=bool, default=False)
    return parser.parse_args()


def has_correct_checksum_md5(filename, hash):
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return str(file_hash.hexdigest()) == hash


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
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7kd5wj7v7p-3.zip'
    zip_filename = os.path.join(root_folder, 'notebooks', 'data', '7kd5wj7v7p-3.zip')
    data_folder = os.path.join(root_folder, 'notebooks', 'data')
    extracted_folder = os.path.join(data_folder, '7kd5wj7v7p-3', 'IXI_sample')

    # Extract if ZIP exists but not folder
    if not os.path.exists(zip_filename):
        # Download if it does not exist
        download_file(url, zip_filename)

    # Check if extracted folder exists
    if os.path.isdir(extracted_folder):
        print(f'Dataset folder already exists in {extracted_folder}')
        return extracted_folder

    assert has_correct_checksum_md5(zip_filename, 'eecb83422a2685937a955251fa45cb03')
    with ZipFile(zip_filename, 'r') as zip_obj:
        zip_obj.extractall(data_folder)

    assert os.path.isdir(extracted_folder)
    return extracted_folder


if __name__ == '__main__':
    args = parse_args()
    root_folder = os.path.abspath(os.path.expanduser(args.root_folder))
    assert os.path.isdir(root_folder), f'Folder does not exist: {root_folder}'

    # Centralized dataset
    centralized_data_folder = download_and_extract_ixi_sample(root_folder)

    # Federated Dataset
    federated_data_folder = os.path.join(root_folder, 'notebooks', 'data', 'Hospital-Centers')
    shutil.rmtree(federated_data_folder, ignore_errors=True)

    csv_global = os.path.join(centralized_data_folder, 'participants.csv')
    allcenters = pd.read_csv(csv_global)

    # Split centers
    center_names = ['Guys', 'HH', 'IOP']
    center_dfs = list()

    for center_name in center_names:
        cfg_folder = os.path.join(args.root_folder, f"{center_name}")
        os.makedirs(cfg_folder, exist_ok=True)
        cfg_file = os.path.join(cfg_folder, f'{center_name.lower()}.ini')

        print(f'Creating node at: {cfg_file}')
        node_component.initiate()
        if node_component.is_component_existing(root_folder):
            print(f"**Warning: component {root_folder} already exists")
        else:
            node_component.initiate(root_folder)

        df = allcenters[allcenters.SITE_NAME == center_name]
        center_dfs.append(df)

        train, test = train_test_split(df, test_size=0.1, random_state=21)

        train_folder = os.path.join(federated_data_folder, center_name, 'train')
        holdout_folder = os.path.join(federated_data_folder, center_name, 'holdout')
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

        train_participants_csv = os.path.join(train_folder, 'participants.csv')
        train.to_csv(train_participants_csv)

        for subject_folder in test.FOLDER_NAME.values:
            shutil.copytree(
                src=os.path.join(centralized_data_folder, subject_folder),
                dst=os.path.join(holdout_folder, subject_folder),
                dirs_exist_ok=True
            )
        test.to_csv(os.path.join(holdout_folder, 'participants.csv'))

    print(f'Centralized dataset located at: {centralized_data_folder}')
    print(f'Federated dataset located at: {federated_data_folder}')

    print()
    print('Please add the data to your nodes executing and using the `ixi-train` tag:')
    for center_name in center_names:
        print(f'\tfedbiomed node --path ./{center_name.lower()} dataset add')

    print()
    print('Then start your nodes by executing:')
    for center_name in center_names:
        print(f'\tfedbiomed node --path ./{center_name.lower()} start')
