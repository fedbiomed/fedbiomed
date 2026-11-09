from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return

        root = Path(self.root)

        if version == "editable":
            pth = root / "_fedbiomed_node_api_editable.pth"

            pth.write_text(
                "import fedbiomed_node_api_editable; "
                f"fedbiomed_node_api_editable.register("
                f"'fedbiomed_node_api', {str(root)!r})\n"
            )

            build_data["force_include_editable"].update(
                {
                    str(root / "editable_import.py"): "fedbiomed_node_api_editable.py",
                    str(pth): "z_fedbiomed_node_api_editable.pth",
                }
            )

        elif version == "standard":
            build_data["force_include"][str(root / "__init__.py")] = (
                "fedbiomed_node_api/__init__.py"
            )
