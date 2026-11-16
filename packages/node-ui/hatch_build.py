import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        super().initialize(version, build_data)

        if self.target_name != "wheel":
            return

        ui_dir = Path(self.root) / "ui"
        build_dir = ui_dir / "build"
        index_file = build_dir / "index.html"

        skip_build = os.environ.get("FBM_SKIP_FRONTEND_BUILD", "").lower() in {
            "1",
            "true",
            "yes",
        }

        if not index_file.is_file() or not skip_build:
            if skip_build:
                logger.warning(
                    "Frontend build was skipped by configuration, "
                    "but assets are missing. Building frontend anyway."
                )
            else:
                logger.info("Building node frontend")

            yarn = shutil.which("yarn")
            if yarn is None:
                raise RuntimeError(
                    "Yarn is required: frontend assets are missing "
                    "or a frontend rebuild was requested."
                )

            for attempt in range(3):
                try:
                    subprocess.run(
                        [yarn, "install"],
                        cwd=ui_dir,
                        check=True,
                    )
                    subprocess.run(
                        [yarn, "build"],
                        cwd=ui_dir,
                        check=True,
                    )
                    break
                except subprocess.CalledProcessError:
                    if attempt == 2:
                        raise
                    time.sleep(5)

        if not index_file.is_file():
            raise RuntimeError("Frontend build did not produce ui/build/index.html")

        build_data["force_include"][str(build_dir)] = "fedbiomed_gui/ui/build"
