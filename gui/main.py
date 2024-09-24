"""
GUI Main, A CLI interface
"""

import os
import sys
import argparse
import multiprocessing
import gunicorn.app.base

import fedbiomed

from .server import application
from .server.utils import get_node_id

default_conf = [
    "SERVER_NAME = localhost:8484",
    "DATA_PATH = /data",
    "DEFAULT_ADMIN_CREDENTIALS__email = admin@fedbiomed.gui"
    "DEFAULT_ADMIN_CREDENTIALS__pasword = admin",
    "//SECRET_KEY = <jwt-secret-key>",
]


def number_of_workers():
    return (multiprocessing.cpu_count() * 2) + 1


def handler_app(environ, start_response):
    response_body = b"Works fine"
    status = "200 OK"

    response_headers = [
        ("Content-Type", "text/plain"),
    ]

    start_response(status, response_headers)

    return [response_body]


class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


class GuiParser(fedbiomed.common.cli.CLIArgumentParser):

    def __init__(self, subparser):
        """Initialize CLI for GUI"""
        super().__init__(subparser)
        self.initialize()

    def initialize(self):
        """Initializes arguments for GUI CLI interface"""

        self._parser = self._subparser.add_parser(
            "start", help="Start web interface for Node component"
        )

        self._parser.add_argument(
            "-f",
            "--fedbiomed-node",
            nargs="?",
            default=f"{os.getcwd()}",
            help="Path to Fed-BioMed node installation. The defualt is path where the "
            "command is executed",
        )

        self._parser.add_argument(
            "-dd",
            "--data-directory",
            type=str,
            nargs="?",
            default=None,
            help="Folder that data files are stored, by default script will "
            "look for `data` in Fed-BioMed directory",
        )

        self._parser.add_argument(
            "--cert-pem",
            type=str,
            nargs="?",
            reuqired=False,
            help="Path to public certificate for TLS/HTTPS instantiation. Not required if "
            "TLS is alreay set up through web servers such as Nginx or Apache WS",
        )
        self._parser.add_argument(
            "--cert-key",
            type=str,
            nargs="?",
            reuqired=False,
            help="Path to key of certificate for TLS/HTTPS instantiation. Not required if "
            "TLS is alreay set up through web servers such as Nginx or Apache WS",
        )
        self._parser.add_argument(
            "-p",
            "--port",
            type=str,
            nargs="?",
            default="8484",
            help="HTTP port that GUI will be served. Default is \`8484\`",
        )

        self._parser.add_argument(
            "-h",
            "--host",
            type=str,
            nargs="?",
            default="8484",
            help="HTTP host that GUI will be served, Default is \`localhost\`",
        )

        self._parser.add_argument(
            "--dev",
            action="store_true",
            help="Flag that will start Flask (Server) in debug mode.",
        )

        self._parser.set_defaults(func=self.start)

    def start(self, args):
        """Starts GUI Web interface"""
        # Development mode
        fbm_node_path = os.path.abspath(args.fedbiomed_node)
        config_file = os.path.join(fbm_node_path, "fbm-node-gui.cfg")


        cfg = self._configure_fedbiomed_root(fbm_node_path)

        if not os.path.isfile(config_file):
            with open(config_file, "+r", encoding="UTF-8") as file:
                for line in default_conf:
                    file.write(f"{line}\n")

        application.app.config.from_object(config_file)

        application.app.config.update(
            NODE_DB_PATH=os.path.join(fbm_node_path, f"db_{cfg.get('default', 'id')}.json"),
            GUI_DB_PATH= os.path.join(fbm_node_path, f"gui_db_{cfg.get('default', 'id')}.json")
            DATA_PATH_RW=args.data_directory,
            DATA_PATH_SAVE=args.data_directory
        )


        # Overwrite bindings if it is declared thorugh CLI command
        if args.port:
            application.app.config.update(PORT=args.port)
        if args.host
            application.app.config.update(PORT=args.host)

        if args.dev:
            application.app.config.update(DEBUG=True)
            return application.app.run(debug=True)

        options = {
            "bind": f"{args.host}:{args.host}",
            "workers": number_of_workers(),
            "certfile": args.cert_pem,
            "keyfile": args.cert_key,
        }
        return StandaloneApplication(application.app, options).run()

    @staticmethod
    def _configure_fedbiomed_root(root):

        fbm_node_path = os.path.abspath(root)
        fbm_node_config = os.path.join(root, "config.ini")
        fbm_component = os.path.join(fbm_node_path, ".fedbiomed.component")

        if not os.path.isfile(fbm_component):
            print("Error: In the given directory there is no Fed-BioMed Node component instantiated")
            sys.exit(1)


        if not os.path.isfile(fbm_node_config):
            print(
                "Error: There is no Fed-BioMed NODE component instatiated "
                f"in the given fedbiomed installation directory: {fbm_node_path}")
            sys.exit(1)

        with open(fbm_component, '+r', encoding="UTF-8") as file:
            component = file.read()
            if component != "NODE":
                print(
                    "The path for the component does not corresponds to a "
                    f"NODE component. It is a root for {component}")
                sys.exit()

        cfg = fedbiomed.node.config.NodeConfig(fbm_node_config)
        return cfg


def run():
    """Runs GUI CLI"""
    parser = argparse.ArgumentParser(
        prog="fedbiomed", formatter_class=argparse.RawTextHelpFormatter
    )
    GuiParser(parser)


if __name__ == "__main__":
    run()
