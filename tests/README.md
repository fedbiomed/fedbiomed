# Tests

Unit tests live with the package whose code they exercise:

| Location | Scope |
| --- | --- |
| `packages/fedbiomed/tests` | Core, node, researcher and the CLI GUI launcher |
| `packages/node-api/tests` | Node API application and routes |
| `packages/node-ui/tests` | GUI application and asset serving |
| `tests/compatibility` | Distribution metadata, builds and package imports |
| `tests/end2end` | End-to-end federated workflows |

With the packages and test dependencies installed, run from the repository root:

```sh
python -m pytest packages/fedbiomed/tests
python -m pytest packages/node-api/tests
python -m pytest packages/node-ui/tests
python -m pytest tests/compatibility
```

Each command discovers the configuration for its test directory automatically.
To run all package tests together, explicitly select the core configuration
because pytest uses one configuration per invocation:

```sh
python -m pytest -c packages/fedbiomed/pyproject.toml packages/fedbiomed/tests packages/node-api/tests packages/node-ui/tests
```

Compatibility tests use their own configuration and do not load runtime test
fixtures. End-to-end tests retain `tests/end2end/pytest.ini`; see their README
for the required services and datasets.
