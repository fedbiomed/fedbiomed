# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Researcher CLI """
import subprocess
import importlib


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

        command = ["jupyter", "notebook"]
        command = [*command, *options]
        process = subprocess.Popen(command)

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

            def import_environ(self) -> 'fedbiomed.researcher.environ.Environ':
                """Import environ"""
                return importlib.import_module("fedbiomed.researcher.environ").environ


        # Config parameter is not necessary. Python client (user in jupyter notebook)
        # will always use default config file which is `researcher_config`
        # However, this argument will play important role once researcher back-end (orhestrator)
        # and researcher is seperated
        self._parser.add_argument(
            "--config",
            "-cf",
            nargs="?",
            action=ConfigNameActionResearcher,
            default="researcher_config.ini",
            help="Name of the config file that the CLI will be activated for. Default is 'researcher_config.ini'.")

        super().initialize()


if __name__ == '__main__':
    cli = ResearcherCLI()
    cli.parse_args()
