import sys
import inspect

from IPython.core.magics.code import extract_symbols
from fedbiomed.common.exceptions import FedbiomedError


def get_class_source(cls) -> str:
    """
        Function for getting source of the class. It uses different method for getting source based on
        shel type. IPython,Notebook shells or Python shell

    Args:
        cls: Class whose source code will be extracted

    Raises:
        none
    Return:
        str: Source code of the given class
    """
    # Check ipython status
    status = is_ipython()

    if status:
        file = _get_ipython_class_file(cls)
        codes = "".join(inspect.linecache.getlines(file))
        class_code = extract_symbols(codes, cls.__name__)[0][0]
        return class_code
    else:
        return inspect.getsource(cls)


def is_ipython() -> bool:
    """
    Function that check whether it is executed in ipython kernel or not

    Args:
        (None)

    Raises:
        (None)

    Returns:
        bool: If True python interpreter is IPython
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


def _get_ipython_class_file(cls) -> str:
    """
    Function that gets source of the class which is defined in ZMQInteractiveShell or
    TerminalInteractiveShell

    Args:
        cls (python class): Python class object defined on the IPython kernel

    Returns:
        str: Returns file path of Jupyter cell. On IPython's interactive shell, it returns cell ID
    """

    if not inspect.isclass(cls):
        raise FedbiomedError(f'The argument `cls` should be a python class ')

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
