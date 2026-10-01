"""GUI launcher, including optional source rebuilding."""

import shutil
import subprocess
from pathlib import Path

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed_node_api.cli import run_launcher, serve


def assets_directory():
    return Path(__file__).parent / "ui" / "build"


def require_assets(assets):
    if not (assets / "index.html").is_file():
        raise FedbiomedError(
            f"GUI frontend assets are missing in {assets}. "
            "From a source checkout run fedbiomed-gui --recreate "
            "(requires Node.js/Yarn), or reinstall a GUI distribution with prebuilt assets."
        )


def launch(args, config):
    assets = assets_directory()
    if args.recreate:
        ui = assets.parent
        if not (ui / "package.json").is_file():
            raise FedbiomedError(
                f"Cannot rebuild the GUI: its sources are not in {ui}."
            )
        yarn = shutil.which("yarn")
        if yarn is None:
            raise FedbiomedError("NodeJS `yarn` is required to rebuild the GUI.")
        try:
            subprocess.run([yarn, "install"], cwd=ui, check=True)
            subprocess.run([yarn, "build"], cwd=ui, check=True)
        except subprocess.CalledProcessError as exc:
            raise FedbiomedError(
                f"Rebuilding the GUI failed: `yarn {exc.cmd[1]}` exited with code {exc.returncode}."
            ) from exc
    require_assets(assets)
    serve(args, config, "fedbiomed_gui.server.wsgi:app")


def run(argv=None):
    run_launcher(launch, gui=True, argv=argv)


if __name__ == "__main__":
    run()
