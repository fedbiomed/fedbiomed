# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Researcher CLI """
import subprocess
import importlib
import os

from typing import List, Dict

from fedbiomed.common.cli import CommonCLI, CLIArgumentParser, ConfigNameAction
from fedbiomed.common.constants import ComponentType


__intro__ = """
   __         _ _     _                          _
  / _|       | | |   (_)                        | |
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` |
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| |    _
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_|   | |
                     _ __ ___  ___  ___  __ _ _ __ ___| |__   ___ _ __
                    | '__/ _ \/ __|/ _ \/ _` | '__/ __| '_ \ / _ \ '__|
                    | | |  __/\__ \  __/ (_| | | | (__| | | |  __/ |
                    |_|  \___||___/\___|\__,_|_|  \___|_| |_|\___|_|
"""


class ResearcherControl(CLIArgumentParser):

    def initialize(self):

        start = self._subparser.add_parser(
            "start", help="Starts Jupyter (server) Notebook for researcher API. The default"
                          "directory will be  notebook directory.")

        start.add_argument(
            "--directory",
            "-dir",
            type=str,
            nargs="?",
            required=False,
            help="The directory where jupyter notebook will be started.")
        start.set_defaults(func=self.start)


    def start(self, args):
        """Starts jupyter notebook"""


        options = ['--NotebookApp.use_redirect_file=false']

        if args.directory:
            options.append(f"--notebook-dir={args.directory}")

        current_env = os.environ.copy()
        #comp_root = os.environ.get("FBM_RESEARCHER_COMPONENT_ROOT", None)
        command = ["jupyter", "notebook"]
        command = [*command, *options]
        process = subprocess.Popen(command, env=current_env)

        try:
            process.wait()
        except KeyboardInterrupt:
            try:
                process.terminate()
            except Exception:
                pass
            process.wait()


class ResearcherCLI(CommonCLI):
    """Researcher CLI"""

    _arg_parsers_classes: List[type] = [
        ResearcherControl,
    ]
    _arg_parsers: Dict[str, CLIArgumentParser] = {}


    def __init__(self):
        super().__init__()
        self.description = f"{__intro__}\nA CLI app for fedbiomed researchers."
        self.initialize()

    def initialize(self):
        """Initializes Researcher CLI"""


        class ConfigNameActionResearcher(ConfigNameAction):
            _this = self
            _component = ComponentType.RESEARCHER

            def import_environ(self, config_file: str | None = None) -> 'fedbiomed.researcher.environ.Environ':
                """Import environ"""
                if config_file:
                    os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = os.path.abspath(config_file)
                else:
                    print("Component is not specified: Using 'fbm-researcher' in current working directory")
                    os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = \
                        os.path.join(os.getcwd(), 'fbm-researcher')

                return importlib.import_module("fedbiomed.researcher.environ").environ


        # Config parameter is not necessary. Python client (user in jupyter notebook)
        # will always use default config file which is `researcher_config`
        # However, this argument will play important role once researcher back-end (orhestrator)
        # and researcher is seperated
        self._parser.add_argument(
            "--config",
            "-c",
            nargs="?",
            action=ConfigNameActionResearcher,
            default="config_researcher.ini",
            help="Name of the config file that the CLI will be activated for. Default is 'config_researcher.ini'.")

        super().initialize()


if __name__ == '__main__':
    cli = ResearcherCLI()
    cli.parse_args()
