import importlib.abc
import importlib.util
import sys
from pathlib import Path


class EditablePackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package_name, package_root):
        self.package_name = package_name
        self.package_root = Path(package_root)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.package_name:
            return None

        init_file = self.package_root / "__init__.py"

        return importlib.util.spec_from_file_location(
            fullname,
            init_file,
            submodule_search_locations=[str(self.package_root)],
        )


def register(package_name, package_root):
    for finder in sys.meta_path:
        if (
            isinstance(finder, EditablePackageFinder)
            and finder.package_name == package_name
        ):
            return

    sys.meta_path.insert(0, EditablePackageFinder(package_name, package_root))
