# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


def is_ipython() -> bool:
    """
    Function that checks whether the codes (function itself) is executed in ipython kernel or not

    Returns:
        True, if python interpreter is IPython
    """

    ipython_shells = ['ZMQInteractiveShell', 'TerminalInteractiveShell', 'Shell']
    try:
        shell = get_ipython().__class__.__name__
        if shell in ipython_shells:
            return True
        else:
            return False
    except NameError:
        return False
