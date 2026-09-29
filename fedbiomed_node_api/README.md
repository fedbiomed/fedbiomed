# Node HTTP API

The API backend builds as the separate `fedbiomed-node-api` distribution and depends on the matching `fedbiomed` version. It doesn't require the frontend packages or assets from the `gui` package.

Build from the repository root with `pdm build -p fedbiomed_node_api` or with `pdm build -p local` to install `node`, `node-api` and the node `gui` packages and their dependencies all in one command.

No frontend assets or Yarn are required for this package's build. For local
editable installation and PDM group selection, see the
[development guide](../docs/developer/development-environment.md#using-pdm).

Run the API without frontend assets against an existing node:

```sh
FBM_NODE_COMPONENT_ROOT=/path/to/node DATA_PATH=/path/to/data \
  python -m gunicorn --workers 1 --bind 127.0.0.1:8484 fedbiomed_node_api.wsgi:app
```

The existing `fedbiomed node ... gui` command and
`fedbiomed_gui.server.wsgi:app` continue to serve both the GUI and API.

Existing `config_gui.ini` files, GUI environment variables, user databases, and
HTTP routes remain compatible. The default configuration template that was previously in the `fedbiomed_gui` package, is now in API `fedbiomed_node_api`.
