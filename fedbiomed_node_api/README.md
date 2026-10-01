# Node HTTP API

The API backend builds as the separate `fedbiomed-node-api` distribution and depends on the matching `fedbiomed` version. It doesn't require the frontend packages or assets from the `gui` package.

Build from the repository root with `pdm build -p fedbiomed_node_api` or with `pdm build -p local` to install `node`, `node-api` and the node `gui` packages and their dependencies all in one command.

No frontend assets or Yarn are required for this package's build. For local
editable installation and PDM group selection, see the
[development guide](../docs/developer/development-environment.md#using-pdm).

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
