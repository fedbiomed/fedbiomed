# Fed-BioMed Node GUI

This distribution contains the GUI server and React frontend. The HTTP API
belongs to `fedbiomed-node-api`; node and dataset operations belong to `fedbiomed`.

Build from the repository root with `pdm build -p fedbiomed_gui`. The GUI build
requires Yarn and compiles the frontend. To package an existing `ui/build`
directory instead, set `FBM_SKIP_FRONTEND_BUILD=1`.

This package depends on the matching `fedbiomed-node-api` version. Core's `gui`
extra selects this distribution. Until it is published, install locally from
the repository root using `pdm sync -G gui -G local`. The `local` development
group installs the sibling packages as editable projects; published metadata
contains regular versioned dependencies without local paths.

Start the GUI and API together:

```sh
fedbiomed-gui --path /path/to/node --data-folder /path/to/data
```

`fedbiomed node --path /path/to/node gui start` remains a compatibility wrapper.
Both commands accept `--host`, `--port`, `--development`, `--debug`,
`--cert-file`, `--key-file`, and `--recreate`. HTTPS requires both certificate
and key files. `--development` uses Flask; the default uses Gunicorn.

Missing frontend assets stop GUI startup with build instructions. From a source
checkout use `fedbiomed-gui --path /path/to/node --recreate` to build them with
Yarn. A wheel without sources must be replaced with a distribution containing
prebuilt assets. API-only startup does not need these files.
