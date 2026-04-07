# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Tkinter import compatibility for Fed-BioMed CLI utilities.

Centralizes tkinter imports and exposes an `available` flag so callers
can guard GUI operations without handling ImportError themselves.
"""

from types import ModuleType
from typing import Optional

# None if tkinter is unavailable on this platform
filedialog: Optional[ModuleType] = None
messagebox: Optional[ModuleType] = None
available: bool = False

try:
    import tkinter._tkinter as _tkinter_internal
    import tkinter.filedialog as _filedialog
    import tkinter.messagebox as _messagebox

    filedialog = _filedialog
    messagebox = _messagebox
    available = True

    class TclError(_tkinter_internal.TclError):
        pass

except ImportError:

    class TclError(Exception):  # type: ignore[no-redef]
        """Raised in place of tkinter.TclError when tkinter is not installed."""
