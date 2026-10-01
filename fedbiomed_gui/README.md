# Fed-BioMed Node GUI

This distribution contains the GUI server and React frontend. The HTTP API
belongs to `fedbiomed-node-api`; node and dataset operations belong to `fedbiomed`.

Build from the repository root with `pdm build -p fedbiomed_gui` or with `pdm build -p local` to install `node`, `node-api` and the node `gui` packages and their dependencies all in one command.. The GUI build requires Yarn and compiles the frontend. To package an existing `ui/build`
directory instead, set `FBM_SKIP_FRONTEND_BUILD=1`.

This package depends on the matching `fedbiomed-node-api` version, which depends
on core. Core's `gui` extra selects this distribution. See the
[development guide](../docs/developer/development-environment.md#using-pdm)
for local editable installation and dependency groups.

Start the GUI and API together:

```sh
fedbiomed-gui --path /path/to/node --data-folder /path/to/data
```

or using the command from previous versions:

```sh
pdm run fedbiomed node --path /path/to/node gui start
```

Both commands accept `--host`, `--port`, `--development`, `--debug`,
`--cert-file`, `--key-file`, and `--recreate`. HTTPS requires both certificate
and key files. `--development` uses Flask; the default uses Gunicorn.

The default host and port for the GUI is `localhost:8484`. 

Without TLS options, the server uses HTTP. For HTTPS, provide a PEM-encoded
server certificate (including its chain when applicable) and its matching
PEM-encoded private key:

```sh
fedbiomed-node-api --path /path/to/node \
  --cert-file /path/to/server-cert.pem --key-file /path/to/server-key.pem
```

Both the public certificate and the private key files are required together and must already be generated. 
These options configure HTTPS for the API/GUI server; they do not configure node–researcher gRPC mutual
authentication. The same options work with `fedbiomed-node-api` and core `fedbiomed` commands as well.

Missing frontend assets stop GUI startup with build instructions. From a source
checkout use `fedbiomed-gui --path /path/to/node --recreate` to build them with
Yarn. A wheel without sources must be replaced with a distribution containing
prebuilt assets. API-only startup does not need these files.
