"""Package frontend assets built explicitly before distribution creation."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        super().initialize(version, build_data)
        # Fresh checkouts must remain installable for --recreate and development.
        if self.target_name == "wheel" and version == "editable":
            return

        assets = Path(self.root) / "ui" / "build"
        if not (assets / "index.html").is_file() or not any(assets.rglob("*.js")):
            raise RuntimeError(
                f"GUI frontend assets are missing or incomplete in {assets}. "
                "From the repository root, run: "
                "cd fedbiomed_gui/ui && yarn install --frozen-lockfile && yarn build. "
                "Then return to the repository root and run pdm build -p fedbiomed_gui. "
                "Released source archives must already contain the built frontend."
            )
