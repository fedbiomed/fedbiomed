# Node HTTP API

The API backend lives here independently of the React frontend. It is still
shipped in the existing Fed-BioMed distribution during the packaging migration;
install `fedbiomed[api]` to obtain its Python dependencies. The existing
`fedbiomed[gui]` extra remains a compatibility alias for these dependencies.

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
to the API. Separate distributions and installation extras are a later change.
