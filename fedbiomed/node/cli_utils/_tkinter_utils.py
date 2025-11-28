# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Simple tkinter import compatibility for Fed-BioMed CLI utilities.

Centralizes tkinter imports in one place and exposes an 'available' flag.
Each file can use try/catch patterns based on this flag.
"""

# Try to import tkinter modules
try:
    import tkinter.filedialog
    import tkinter.messagebox
    from tkinter import _tkinter

    available = True
except (ImportError, ModuleNotFoundError):
    tkinter = None
    available = False

# Export for convenience when available
if available:
    filedialog = tkinter.filedialog
    messagebox = tkinter.messagebox
    TclError = _tkinter.TclError
else:
    filedialog = None
    messagebox = None

    class TclError(Exception):
        """Fallback TclError when tkinter is not available."""

        pass
