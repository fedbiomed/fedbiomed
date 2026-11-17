# Node HTTP API

The API backend builds as the separate `fedbiomed-node-api` distribution and
depends on the matching `fedbiomed` version. Core's `node-api` extra selects it; `gui` selects the GUI distribution, which also depends on the API.

Build from the repository root with `pdm build -p fedbiomed_node_api` or with `pdm build -p local` to install `node`, `node-api` and the node `gui` packages and their dependencies all in one command.

No frontend assets or Yarn are required for this package's build. For local
editable installation and PDM group selection, see the
[development guide](../docs/developer/development-environment.md#using-pdm).

Run the API without frontend assets against an existing node:

```sh
fedbiomed node --path /path/to/node api start --data-folder /path/to/data
```

The default is HTTP on `localhost:8484`. Use `--host` and `--port` to change
the address, or supply both `--cert-file` and `--key-file` for HTTPS.
`--development` selects Flask; add `--debug` to enable its debugger and reloader.
If the API package is missing, the CLI prints the matching `fedbiomed[node-api]`
installation command. API-only startup does not require the GUI or Yarn.

The existing `fedbiomed node ... gui` command and
`fedbiomed_gui.server.wsgi:app` continue to serve both the GUI and API.

Existing `config_gui.ini` files, GUI environment variables, user databases, and
HTTP routes remain compatible. The default configuration template that was previously in the `fedbiomed_gui` package, is now in API `fedbiomed_node_api`.
