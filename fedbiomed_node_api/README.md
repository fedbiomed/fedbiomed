# Node HTTP API

The API backend lives here independently of the React frontend and builds as
the separate `fedbiomed-node-api` distribution. Build from the repository root
with `pdm build -p fedbiomed_node_api`; no frontend tools or assets are needed.

Package dependency declarations and installation extras are introduced in the
next migration step. Until then, these build artifacts are for packaging
verification, not standalone installation.

Run the API without frontend assets against an existing node:

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
to the API. Installation extras are a later change.
