import pkgutil
from typing import Dict

import flamby.datasets as fsets


def discover_flamby_datasets() -> Dict[int, str]:
    """Automatically discover the available Flamby datasets based on the contents of the flamby.datasets module.

    Returns:
        a dictionary {index: dataset_name} where index is an int and dataset_name is the name of a flamby dataset
        represented as a str.

    """
    dataset_list = [name for _, name, ispkg in pkgutil.iter_modules(fsets.__path__) if ispkg]
    return {i: name for i, name in enumerate(dataset_list)}
