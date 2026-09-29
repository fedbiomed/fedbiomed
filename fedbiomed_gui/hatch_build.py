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
        # Code in this function will run before building
        super().initialize(version, build_data)

        if os.environ.get("FBM_SKIP_FRONTEND_BUILD", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            logger.info("Skipping node front-end build")
            return

        logger.info("Building node front-end")
        yarn = shutil.which("yarn")

        if yarn is None:
            raise RuntimeError(
                "NodeJS `yarn` is required for building Fed-BioMed front-end application"
            )

        ui_dir = Path(self.root) / "ui"
        for attempt in range(3):
            try:
                logger.info(
                    "### Yarn: Installation front-end dependencies to prepare build.\n"
                )
                subprocess.run([yarn, "install"], cwd=ui_dir, check=True)
                logger.info("\n### Yarn: Building front-end application run.\n")
                subprocess.run([yarn, "build"], cwd=ui_dir, check=True)
            except subprocess.CalledProcessError:
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise
            else:
                break
