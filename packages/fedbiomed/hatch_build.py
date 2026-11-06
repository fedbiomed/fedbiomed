"""Carry repository assets into standalone source archives and wheels."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        root = Path(self.root)
        readme = root / self.config["readme"]
        if not readme.is_file():
            readme = root.parents[1] / self.config["readme"]
        metadata["readme"] = {
            "text": readme.read_text(encoding="utf-8"),
            "content-type": "text/markdown",
        }


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        # A source archive carries its own assets; a checkout uses repo files.
        source = root if (root / "notebooks").is_dir() else root.parents[1]
        if self.target_name == "sdist":
            for name, destination in self.config["sdist-files"].items():
                build_data["force_include"][str(source / name)] = destination
        for name, destination in self.config["shared-data"].items():
            path = source / name
            if not path.is_dir():
                raise RuntimeError(f"Missing shared package assets: {path}")
            if self.target_name == "sdist":
                build_data["force_include"][str(path)] = name
            else:
                build_data.setdefault("shared_data", {})[str(path)] = destination
