#!/usr/bin/env python
import hashlib
import os
import pathlib
import uuid

import requests
from tqdm import tqdm
import argparse
from zipfile import ZipFile
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tinydb import TinyDB, Query

FEDBIOMED_ROOT = str(pathlib.Path(__file__).parent.resolve().parent)
print('Root for Fed-BioMed:', FEDBIOMED_ROOT)

config_file = """
[default]
node_id = CENTER_ID
uploads_url = http://localhost:8844/upload/

[mqtt]
broker_ip = localhost
port = 1883
keep_alive = 60

[security]
hashing_algorithm = SHA256
allow_default_models = True
model_approval = False
"""


def parse_args():
    parser = argparse.ArgumentParser(description='IXI Sample downloader and splitter')
    parser.add_argument('-f', '--root_folder', required=True, type=str)

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
    url = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/7kd5wj7v7p-3.zip'
    zip_filename = os.path.join(root_folder, '7kd5wj7v7p-3.zip')
    data_folder = os.path.join(root_folder, '7kd5wj7v7p-3', 'IXI_sample')

    # Check if extracted folder exists
    if os.path.isdir(data_folder):
        print(f'Dataset folder already exists in {data_folder}')
        return data_folder

    # Extract if ZIP exists but not folder
    if not os.path.exists(zip_filename):
        # Download if it does not exist
        download_file(url, zip_filename)

    assert has_correct_checksum_md5(zip_filename, 'eecb83422a2685937a955251fa45cb03')
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
    federated_data_folder = os.path.join(root_folder, 'UniCancer-Centers')

    csv_global = os.path.join(centralized_data_folder, 'participants.csv')
    allcenters = pd.read_csv(csv_global)

    # Split centers
    center_names = ['Guys', 'HH', 'IOP']
    center_dfs = list()

    for center_name in center_names:
        cfg_folder = os.path.join(FEDBIOMED_ROOT, 'etc')
        os.makedirs(cfg_folder, exist_ok=True)
        cfg_file = os.path.join(cfg_folder, f'{center_name.lower()}.ini')

        print(f'Creating node at: {cfg_file}')
        with open(cfg_file, 'w') as f:
            f.write(config_file.replace('CENTER_ID', center_name))

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

        # Populate node
        print('Populating nodes...')
        db_folder = os.path.join(FEDBIOMED_ROOT, 'var')
        os.makedirs(db_folder, exist_ok=True)
        db_file = os.path.join(db_folder, f'db_{center_name}.json')
        db = TinyDB(db_file)
        db.insert({
            "name": "IXI",
            "data_type": "bids",
            "tags": ["bids-train"],
            "description": "IXI",
            "shape": {
                "label": [83, 44, 55],
                "T1": [83, 44, 55], "T2": [83, 44, 55],
                "demographics": [len(train), 13],
                "num_modalities": 3},
            "path": train_folder,
            "dataset_id": f"dataset_{uuid.uuid4()}",
            "dtypes": [],
            "dataset_parameters": {
                "tabular_file": train_participants_csv,
                "index_col": 14
            }})

    print(f'Centralized dataset located at: {centralized_data_folder}')
    print(f'Federated dataset located at: {federated_data_folder}')

    print()
    print(f'Please start your nodes executing:')
    for center_name in center_names:
        print(f'\t./scripts/fedbiomed_run node config {center_name.lower()}.ini start')
