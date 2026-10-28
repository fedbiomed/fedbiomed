# Node HTTP API

This package depends on the matching `fedbiomed` version and the backend web
dependencies. Install a released version with `pip install "fedbiomed[node-api]"`; core's `node-api` extra selects this distribution. It doesn't require the frontend packages or assets from the `gui` package.

See the [user guide](../docs/user-guide/nodes/node-api.md) for curl examples and
Postman setup, and the [API reference](../docs/user-guide/nodes/node-api-reference.md)
for methods and request bodies. The packaged [openapi.json](openapi.json) is also
served at `/openapi.json` without authentication in API-only and GUI mode.

For development, install from the repository using `pdm sync --prod -G node-api`.
The checked-in lockfile selects the local editable API package without the GUI.
The development-only `local` group supplies sibling paths when regenerating
the lockfile with `pdm lock -G :all --update-reuse`. For details on local editable installation and PDM group selection, see the [development guide](../docs/developer/development-environment.md#using-pdm).

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

Both the public certificate and the private key files are required together and must already be generated. These options configure
HTTPS for the API/GUI server; they do not configure node–researcher gRPC mutual
authentication. The same options work with `fedbiomed-gui` and core `fedbiomed` commands as well.

For direct WSGI deployment, the existing entry point remains available:

```sh
FBM_NODE_COMPONENT_ROOT=/path/to/node DATA_PATH=/path/to/data \
  python -m gunicorn --workers 1 --bind 127.0.0.1:8484 fedbiomed_node_api.wsgi:app
```

The existing `fedbiomed node ... gui` command and
`fedbiomed_gui.server.wsgi:app` continue to serve both the GUI and API.

Existing `config_gui.ini` files, GUI environment variables, user databases, and
HTTP routes remain compatible. The default configuration template that was previously in the `fedbiomed_gui` package, is now in API `fedbiomed_node_api`.
