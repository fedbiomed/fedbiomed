"""Options shared by standalone launchers and core compatibility wrappers.

Keep this module independent of optional web packages so core can show command
help and installation guidance even when those packages are absent.
"""


def add_server_arguments(parser, *, gui=False):
    parser.add_argument("--data-folder", "-df", default="")
    parser.add_argument(
        "--cert-file",
        "-cf",
        help="PEM server certificate for HTTPS (requires --key-file)",
    )
    parser.add_argument(
        "--key-file",
        "-kf",
        help="Matching PEM private key for HTTPS (requires --cert-file)",
    )
    parser.add_argument("--port", "-p", default="8484")
    parser.add_argument("--host", "-ho", default="localhost")
    parser.add_argument(
        "--debug",
        "-dbg",
        action="store_true",
        help="Enable Flask debug mode; with --development, also enable the debugger and reloader",
    )
    parser.add_argument(
        "--development",
        "-dev",
        action="store_true",
        help="Use the Flask development server",
    )
    if gui:
        parser.add_argument(
            "--recreate",
            "-rc",
            action="store_true",
            help="Rebuild frontend sources with yarn install and yarn build",
        )
