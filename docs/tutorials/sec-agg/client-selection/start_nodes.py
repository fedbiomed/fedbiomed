import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run multiple Fedbiomed nodes.')
parser.add_argument('--num_nodes', type=int, default=180, help='Number of nodes to run')
args = parser.parse_args()

num_nodes = args.num_nodes

FEDBIOMED_DIR = os.environ.get('FEDBIOMED_DIR')
if FEDBIOMED_DIR is None:
    raise ValueError('Please define the environment variable FEDBIOMED_DIR by running the command: export FEDBIOMED_DIR=$(pwd) in the fedbiomed root directory.')

def execute_command(config_name):
    # Construct the command
    command = f'{FEDBIOMED_DIR}/scripts/fedbiomed_run node --config {config_name} start'
    
    # Print the command for debugging
    print(f'Executing: {command}')
    
    # Execute the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Return the result
    return (config_name, result.stdout, result.stderr)

config_names = [f'config-n{i}.ini' for i in range(num_nodes)]

# Create a ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_nodes) as executor:
    # Submit all tasks
    futures = [executor.submit(execute_command, config_name) for config_name in config_names]
    
    # Process results as they complete
    for future in as_completed(futures):
        config_name, stdout, stderr = future.result()
        print(f'Config: {config_name}')
        print(f'Output:\n{stdout}')
        print(f'Error:\n{stderr}')

print('All commands executed.')
