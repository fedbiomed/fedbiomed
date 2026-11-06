"""Carry repository assets into standalone source archives and wheels."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface


def readme_path(root):
    root = Path(root)
    # Extracted source archives have their own README; the checkout uses root's.
    bundled = root / "README.md"
    return bundled if bundled.is_file() else root.parents[1] / "README.md"


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        metadata["readme"] = {
            "text": readme_path(self.root).read_text(encoding="utf-8"),
            "content-type": "text/markdown",
        }


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        # A source archive carries its own assets; a checkout uses repo files.
        source = root if (root / "notebooks").is_dir() else root.parents[1]
        if self.target_name == "sdist":
            build_data["force_include"][str(readme_path(root))] = "README.md"
        for name in ("docs/tutorials", "notebooks"):
            path = source / name
            if not path.is_dir():
                raise RuntimeError(f"Missing shared package assets: {path}")
            if self.target_name == "sdist":
                build_data["force_include"][str(path)] = name
            else:
                build_data.setdefault("shared_data", {})[str(path)] = (
                    f"share/fedbiomed/{name}"
                )
