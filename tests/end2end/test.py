from execution import (
    shell_process,
    collect,
    execute_in_paralel,
)

from helpers import (
    create_component,
    add_dataset_to_node,
    clear_component_data,
    execute_script
)


from constants import CONFIG_PREFIX

from fedbiomed.common.constants import ComponentType


print("Executing notebook file getting started")
execute_script("./notebooks/101_getting-started.ipynb", activate='researcher')
