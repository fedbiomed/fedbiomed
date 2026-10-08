# Node HTTP API

The API backend lives here independently of the React frontend and builds as
the separate `fedbiomed-node-api` distribution. Build from the repository root
with `pdm build -p fedbiomed_node_api`; no frontend tools or assets are needed.

This package depends on the matching `fedbiomed` version and the backend web
dependencies. Install a released version with
`pip install "fedbiomed[node-api]"`; core's `node-api` extra selects
this distribution. For development, install from the repository using
`pdm sync --prod -G node-api`.
The checked-in lockfile selects the local editable API package without the GUI.
The development-only `local` group supplies sibling paths when regenerating
the lockfile with `pdm lock -G :all --update-reuse`.

Run the API without frontend assets against an existing node:

```sh
fedbiomed-node-api --path /path/to/node --data-folder /path/to/data
# Equivalent core command:
fedbiomed node --path /path/to/node api start --data-folder /path/to/data
```

The default bind address is `localhost:8484`; use `--host` and `--port` to
change it. The data folder defaults to the `data` directory in the node's root.
A missing node component is initialized automatically by the standalone launcher.
`--development` selects Flask's development server instead of Gunicorn.
`--debug` enables Flask debug mode and debug-level application logging; with
`--development`, it also enables the debugger and backend code reloader.
Neither flag is required for normal API use.

Without TLS options, the server uses HTTP. For HTTPS, provide a PEM-encoded
server certificate (including its chain when applicable) and its matching
PEM-encoded private key:

```sh
fedbiomed-node-api --path /path/to/node \
  --cert-file /path/to/server-cert.pem --key-file /path/to/server-key.pem
```

Both files are required together and must already exist. These options configure
HTTPS for the API/GUI server; they do not configure node–researcher gRPC mutual
authentication. The same options work with `fedbiomed-gui` and the core wrappers.
The launcher uses the active Python interpreter for both servers.

For direct WSGI deployment, the existing entry point remains available:

```sh
FBM_NODE_COMPONENT_ROOT=/path/to/node DATA_PATH=/path/to/data \
  gunicorn --workers 1 --bind 127.0.0.1:8484 fedbiomed_node_api.wsgi:app
```

The existing `fedbiomed node ... gui` command and
`fedbiomed_gui.server.wsgi:app` continue to serve both the GUI and API.

`fedbiomed_node_api.application.create_app(node_root=None, overrides=None)`
creates an API application with no static routes. Omitting `node_root` uses
`FBM_NODE_COMPONENT_ROOT`, falling back to the working directory. The GUI factory
calls this factory and adds frontend routes. Configuration, databases, managers,
and caches belong to each application; route modules resolve them through the
active Flask application context. Importing the factory does not initialize a node.

Existing `config_gui.ini` files, GUI environment variables, user databases, and
HTTP routes remain compatible. The default configuration template now belongs
to the API.
