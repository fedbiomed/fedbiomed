import flamby.datasets as fsets
import pkgutil

def get_flamby_datasets():
    """Get automatically dataset (dataset name and module name containing the federated class) from the flamby package.

    Returns:
        A tuple containing 2 elements :
            - the first is a dictionary with integers as keys (1 to X, X being the number of flamby datasets)
            and values being the absolute path of the folder which contains the federated class of the dataset
            - the second is also a dictionary, with same keys as the first dictionary, and values being a short name identifying
            each dataset, to display in the interface all the options of flamby datasets in a convenient way in the selection menu.

    """
    prefix = fsets.__name__ + "."
    available_flamby_datasets = {}
    for i, (importer, modname, ispkg) in enumerate(pkgutil.iter_modules(fsets.__path__, prefix), start=1):
        available_flamby_datasets[i] = modname
    valid_flamby_options = {i: val.split(".")[-1][4:] for i, val in available_flamby_datasets.items()}
    return available_flamby_datasets, valid_flamby_options


def get_key_from_value(my_dict: dict, val: str):
    """Get key from value of a particular dictionary, if the searched value is present in it.

    Args:
        my_dict: dictionary
        val: value
    Returns:
        The key as a string if the value is found, or a string indicating that the key doesn't exist.
    """
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"