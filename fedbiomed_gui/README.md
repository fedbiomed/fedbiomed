# Fed-BioMed Node GUI

This distribution contains the GUI server and React frontend. The HTTP API
belongs to `fedbiomed-node-api`; node and dataset operations belong to `fedbiomed`.

Build from the repository root with `pdm build -p fedbiomed_gui` or with `pdm build -p local` to install `node`, `node-api` and the node `gui` packages and their dependencies all in one command.. The GUI build requires Yarn and compiles the frontend. To package an existing `ui/build`
directory instead, set `FBM_SKIP_FRONTEND_BUILD=1`.

This package depends on the matching `fedbiomed-node-api` version, which depends
on core. Core's `gui` extra selects this distribution. See the
[development guide](../docs/developer/development-environment.md#using-pdm)
for local editable installation and dependency groups.

Start the GUI and API together against your node:

```sh
pdm run fedbiomed node --path /path/to/node gui start
```

The default is HTTP on `localhost:8484`. Supply both `--cert-file` and
`--key-file` for HTTPS. See the [GUI guide](../docs/user-guide/nodes/node-gui.md)
for configuration.
