# Fed-BioMed Node GUI

This distribution contains the GUI server and React frontend. The HTTP API
belongs to `fedbiomed-node-api`; node and dataset operations belong to `fedbiomed`.

Build from the repository root with `pdm build -p fedbiomed_gui`. The GUI build
requires Yarn and compiles the frontend. To package an existing `ui/build`
directory instead, set `FBM_SKIP_FRONTEND_BUILD=1`.

Package dependency declarations and installation extras are introduced in the
next migration step. Until then, these build artifacts are for packaging
verification, not standalone installation.
