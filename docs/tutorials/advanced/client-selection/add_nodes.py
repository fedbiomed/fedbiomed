import os
import subprocess
import json

# Define the path to the directory containing CSV files
# get env variable FEDBIOMED_DIR
FEDBIOMED_DIR = os.environ.get('FEDBIOMED_DIR')
# check if FEDBIOMED_DIR is None otherwise raise an error and remind to define export FEDBIOMED_DIR=$(pwd) in fedbiomed root directory
if FEDBIOMED_DIR is None:
    raise ValueError('Please define the environment variable FEDBIOMED_DIR by running the command: export FEDBIOMED_DIR=$(pwd) in the fedbiomed root directory.')
DATA_DIR = f'{FEDBIOMED_DIR}/data/replace-bg/raw/patients'

# List all CSV files in the directory
patient_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
# Check if there are any CSV files
if len(patient_files) == 0:
    raise ValueError('No CSV files found in the directory, please run preprocessing.py to generate the CSV files.')
# order the files
patient_files.sort()
# Iterate over each CSV file
for i, patient_file in enumerate(patient_files):
    print(f'Processing file: {patient_file}')
    file_path = os.path.join(DATA_DIR, patient_file)
    config_name = f'config-n{i}.ini'
    
    # Create the JSON-like structure for the filename
    filename_json = {
        "name": "replace-bg",
        "description": "replace-bg",
        "tags": "replace-bg",
        "data_type": "csv",
        "path": file_path
    }

    dataset_info = {
        "name": "replace-bg",
        "description": "replace-bg",
        "tags": "replace-bg",
        "data_type": "csv",
        "path": file_path
    }

    # Convert the dataset_info to a JSON string
    dataset_info_json = json.dumps(dataset_info)
    # write the dataset_info_json to a file 
    with open(f'{FEDBIOMED_DIR}/data/replace-bg/dataset_info.json', 'w') as f:
        f.write(dataset_info_json)
    
    # Construct the command
    command = f'{FEDBIOMED_DIR}/scripts/fedbiomed_run node --config {config_name} dataset add --file {FEDBIOMED_DIR}/data/replace-bg/dataset_info.json'

    # Print the command for debugging
    print(f'Executing: {command}')
    
    # Execute the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Print the result
    print(f'Output:\n{result.stdout}')
    print(f'Error:\n{result.stderr}')

print('All commands executed.')
