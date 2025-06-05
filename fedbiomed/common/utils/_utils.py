# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import inspect
import importlib.util
import re
import uuid

from collections.abc import Iterable
from typing import Callable, Iterator, List, Optional, Union, Any, Tuple

import torch
import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.ipython import is_ipython




def read_file(path):
    """Read given file

    Args:
        path: Path to file to be read

    Raises:
        FedbiomedError: If the file is not existing or readable.
    """
    try:
        with open(path, "r", encoding="UTF-8") as file:
            content = file.read()
            file.close()
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB627.value}: Can not read file {path}. Error: {e}"
        ) from e

    return content


def get_class_source(cls: Callable) -> str:
    """Get source of the class.

    It uses different methods for getting the class source based on shell type; IPython,Notebook
    shells or Python shell

    Args:
        cls: The class to extract the source code from

    Returns:
        str: Source code of the given class

    Raises:
        FedbiomedError: if argument is not a class
    """

    if not inspect.isclass(cls):
        raise FedbiomedError(f'{ErrorNumbers.FB627.value}: The argument `cls` must be a python class')

    # Check ipython status
    status = is_ipython()

    if status:
        file = get_ipython_class_file(cls)
        codes = "".join(inspect.linecache.getlines(file))

        # Import only on IPython interface
        module = importlib.import_module("IPython.core.magics.code")
        extract_symbols = module.extract_symbols
        print(extract_symbols)
        class_code = extract_symbols(codes, cls.__name__)[0][0]
        return class_code

    return inspect.getsource(cls)


def import_class_object_from_file(module_path: str, class_name: str) -> Tuple[Any, Any]:
    """Import a module from a file and create an instance of a specified class of the module.

    Args:
        module_path: path to python module file
        class_name: name of the class

    Returns:
        Tuple of the module created and the training plan object created

    Raises:
        FedbiomedError: bad argument type
        FedbiomedError: cannot instantiate object
    """
    for arg in [module_path, class_name]:
        if not isinstance(arg, str):
            raise FedbiomedError(f"{ErrorNumbers.FB627.value}: Expected argument type is string but got '{type(arg)}'")

    module, train_class = import_class_from_file(module_path, class_name)

    try:
        train_class_instance = train_class()
    except Exception as e:
        raise FedbiomedError(f"{ErrorNumbers.FB627.value}: Cannot instantiate training plan object: {e}")

    return module, train_class_instance



def import_object(module: str, obj_name: str) -> Any:
    """Imports given object/class from given module


    Args:
        module: Module that the object will be imported from
        obj_name: Name of the object or class defined in the module

    Returns:
        Python object
    """
    try:
        module = importlib.import_module(module)
    except ModuleNotFoundError as exp:
        raise FedbiomedError(f"Specified module is not existing. {exp}") from exp

    try:
        obj_ = getattr(module, obj_name)
    except AttributeError as exp:
        raise FedbiomedError(f"{ErrorNumbers.FB627}, Attribute error while loading the class "
                             f"{obj_name} from {module}. Error: {exp}") from exp


    return obj_

def import_class_from_file(module_path: str, class_name: str) -> Tuple[Any, Any]:
    """Import a module from a file and return a specified class of the module.

    Args:
        module_path: path to python module file
        class_name: name of the class

    Returns:
        Tuple of the module created and the training plan class loaded

    Raises:
        FedbiomedError: bad argument type
        FedbiomedError: cannot load module or class
    """
    if not os.path.isfile(module_path):
        raise FedbiomedError(f"{ErrorNumbers.FB627}: Given path for importing {class_name} is not existing")

    module_base_name = os.path.basename(module_path)
    pattern = re.compile("(.*).py$")

    match = pattern.match(module_base_name)
    if not match:
        raise FedbiomedError(f"{ErrorNumbers.FB627}: File is not a python file.")

    module = match.group(1)
    sys.path.insert(0, os.path.dirname(module_path))


    class_ = import_object(module, class_name)
    sys.path.pop(0)

    return module, class_


def import_class_from_spec(code: str, class_name: str) -> Tuple[Any, Any] :
    """Import a module from a code and extract the code of a specified class of the module.

    Args:
        code: code of the module
        class_name: name of the class

    Returns:
        Tuple of the module created and the extracted class

    Raises:
        FedbiomedError: bad argument type
        FedbiomedError: cannot load module or extract clas
    """

    for arg in [code, class_name]:
        if not isinstance(arg, str):
            raise FedbiomedError(f"{ErrorNumbers.FB627.value}: Expected argument type is string but got '{type(arg)}'")

    try:
        spec = importlib.util.spec_from_loader("module_", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(code, module.__dict__)
    except Exception as e:
        raise FedbiomedError(f"{ErrorNumbers.FB627.value}: Can not load module from given code: {e}")

    try:
        class_ = getattr(module, class_name)
    except AttributeError:
        raise FedbiomedError(f"{ErrorNumbers.FB627.value}: Can not import {class_name} from given code")

    return module, class_


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
        raise FedbiomedError(f'{ErrorNumbers.FB627.value}: {cls} has no attribute `__module__`, source is not found.')


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
        raise FedbiomedError(
            f"{ErrorNumbers.FB627.value}: Converting {type(value)} to python to float is not supported.")

    # if the result is a tensor, convert it back to numpy
    if isinstance(value, torch.Tensor):
        value = value.numpy()

    if isinstance(value, Iterable) and value.size > 1:
        raise FedbiomedError(f"{ErrorNumbers.FB627.value}: Can not convert array-type objects to float.")

    return float(value)


def convert_iterator_to_list_of_python_floats(iterator: Iterator) -> List[float]:
    """Converts numerical values of array-like object to float

    Args:
        iterator: Array-like numeric object to convert numerics to float

    Returns:
        Numerical elements as converted to List of floats
    """

    if not isinstance(iterator, Iterable):
        raise FedbiomedError(f"{ErrorNumbers.FB627.value}: object {type(iterator)} is not iterable")

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
