
from helpers import (
    create_component,
    add_dataset_to_node,
    clear_node_data,
    execute_script,
    collect_output_in_parallel
)


from helpers.constants import CONFIG_PREFIX

from fedbiomed.common.constants import ComponentType


print("Executing notebook file getting started")
execute_script("./notebooks/101_getting-started.ipynb", activate='researcher')
