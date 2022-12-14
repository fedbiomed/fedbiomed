# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import sys
import inspect
from collections.abc import Iterable
from typing import Callable, Iterator, List, Optional, Union
from IPython.core.magics.code import extract_symbols

import torch
import numpy as np
from fedbiomed.common.exceptions import FedbiomedError


def read_file(path):
    """Read given file

    Args:
        path: Path to file to be read

    Raises:
        FedbiomedError: If the file is not existing or readable.
    """
    try:
        with open(path, "r") as file:
            content = file.read()
            file.close()
    except Exception as e:
        raise FedbiomedError(
            f"Can not read file {path}. Error: {e}"
        )
    else:
        return content


def get_class_source(cls: Callable) -> str:
    """Get source of the class.

    It uses different methods for getting the class source based on shell type; IPython,Notebook
    shells or Python shell

    Args:
        cls: The class to extract the source code from

    Return:
        str: Source code of the given class

    Raises:
        FedbiomedError: if argument is not a class
    """

    if not inspect.isclass(cls):
        raise FedbiomedError('The argument `cls` must be a python class')

    # Check ipython status
    status = is_ipython()

    if status:
        file = get_ipython_class_file(cls)
        codes = "".join(inspect.linecache.getlines(file))
        class_code = extract_symbols(codes, cls.__name__)[0][0]
        return class_code
    else:
        return inspect.getsource(cls)


def is_ipython() -> bool:
    """
    Function that checks whether the codes (function itself) is executed in ipython kernel or not

    Returns:
        True, if python interpreter is IPython
    """

    ipython_shells = ['ZMQInteractiveShell', 'TerminalInteractiveShell']
    try:
        shell = get_ipython().__class__.__name__
        if shell in ipython_shells:
            return True
        else:
            return False
    except NameError:
        return False


def get_ipython_class_file(cls: Callable) -> str:
    """Get source file/cell-id of the class which is defined in ZMQInteractiveShell or TerminalInteractiveShell

    Args:
        cls: Python class defined on the IPython kernel

    Returns:
        File path or id of Jupyter cell. On IPython's interactive shell, it returns cell ID
    """

    # Lookup by parent module
    if hasattr(cls, '__module__'):
        object_ = sys.modules.get(cls.__module__)
        # If module has `__file__` attribute
        if hasattr(object_, '__file__'):
            return object_.__file__

        # If parent module is __main__
        for name, member in inspect.getmembers(cls):
            if inspect.isfunction(member) and cls.__qualname__ + '.' + member.__name__ == member.__qualname__:
                return inspect.getfile(member)
    else:
        raise FedbiomedError(f'{cls} has no attribute `__module__`, source is not found.')


def get_method_spec(method: Callable) -> dict:
    """ Helper to get argument specification

    Args:
        method: The function/method to extract argument specification from

    Returns:
         Specification of the method
    """

    method_spec = {}
    parameters = inspect.signature(method).parameters
    for (key, val) in parameters.items():
        method_spec[key] = {
            'name': val.name,
            'default': None if val.default is inspect._empty else val.default,
            'annotation': None if val.default is inspect._empty else val.default
        }

    return method_spec


def convert_to_python_float(value: Union[torch.Tensor, np.integer, np.floating, float, int]) -> float:
    """ Convert numeric types to float

    Args:
        value: value to convert python type float

    Returns:
        Python float
    """

    if not isinstance(value, (torch.Tensor, np.integer, np.floating, float, int)):
        raise FedbiomedError(f"Converting {type(value)} to python to float is not supported.")

    # if the result is a tensor, convert it back to numpy
    if isinstance(value, torch.Tensor):
        value = value.numpy()

    if isinstance(value, Iterable) and value.size > 1:
        raise FedbiomedError("Can not convert array-type objects to float.")

    return float(value)


def convert_iterator_to_list_of_python_floats(iterator: Iterator) -> List[float]:
    """Converts numerical values of array-like object to float

    Args:
        iterator: Array-like numeric object to convert numerics to float

    Returns:
        Numerical elements as converted to List of floats
    """

    if not isinstance(iterator, Iterable):
        raise FedbiomedError(f"object {type(iterator)} is not iterable")

    list_of_floats = []
    if isinstance(iterator, dict):
        # specific processing for dictionaries
        for val in iterator.values():
            list_of_floats.append(convert_to_python_float(val))
    else:
        for it in iterator:
            list_of_floats.append(convert_to_python_float(it))
    return list_of_floats


def compute_dot_product(model: dict, params: dict, device: Optional[str] = None) -> torch.tensor:
    """Compute the dot product between model and input parameters.

    Args:
        model: OrderedDict representing model state
        params: OrderedDict containing correction parameters

    Returns:
        A tensor containing a single numerical value which is the dot product.
    """
    model_p = model.values()
    correction_state = params.values()
    if device is None:
        if model_p:
            device = list(model_p)[0].device
        else:
            # if device is not found, set it to `cpu`
            device = 'cpu'
    dot_prod = sum([torch.sum(m * torch.tensor(p).float().to(device)) for m, p in zip(model_p, correction_state)])
    return dot_prod