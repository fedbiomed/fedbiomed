# Node GUI

The Node GUI provides a browser interface to the same HTTP API available in an
[API-only installation](node-api.md). It manages datasets, training plans and
node configuration.

## Install and start

```sh
pip install "fedbiomed[gui]"
fedbiomed-gui --path /path/to/node
# Equivalent core command:
fedbiomed node --path /path/to/node gui start
```

Choose one command. Both serve the GUI and API at **http://localhost:8484**
using Gunicorn by default. Open this address to log in; see
[default administrator configuration](#default-admin-configuration).
Released packages include the built frontend and need neither Node.js nor Yarn.

`--path` selects the node component directory, including its configuration and
databases. It defaults to `fbm-node` under the current working directory. A missing
component is initialized automatically. Use the same path as your existing node
to manage its datasets and accounts.

**Starting the GUI does not start the federated-learning node.** Start the node
from the Node Management page, or in another terminal:

```sh
fedbiomed node --path /path/to/node start
```

## Server and data paths

```sh
fedbiomed-gui --path /path/to/node --data-folder /path/to/datasets --port 8485
# Equivalent:
fedbiomed node --path /path/to/node gui start --data-folder /path/to/datasets --port 8485
```

The data folder defaults to `data` in the node's root and must exist. Dataset
registration selects files already present on the server; it does not upload
files from your browser. API path arrays are relative to this folder.

Use `--host` to change the listening address. For multiple nodes, use a distinct
node path and port for each server. The API-only and GUI launchers both use port
8484 by default; run only one for a given node unless you explicitly configure
separate ports. The GUI already includes all API endpoints.

## HTTPS configuration

Without TLS options, the server uses HTTP. Supply both an existing PEM server
certificate (including its chain when applicable) and its matching PEM private
key to enable HTTPS:

```sh
fedbiomed-gui --path /path/to/node \
  --cert-file /path/to/server-cert.pem --key-file /path/to/server-key.pem
# Equivalent:
fedbiomed node --path /path/to/node gui start \
  --cert-file /path/to/server-cert.pem --key-file /path/to/server-key.pem
```

Open `https://localhost:8484` (or the hostname covered by your certificate).
The client must trust the certificate's issuer. The same TLS options work with
`fedbiomed-node-api` and `fedbiomed node ... api start`, in both Gunicorn and
Flask development mode. TLS can also terminate at a reverse proxy.

These certificates protect browser/API-client connections to the HTTP server.
They are **separate from node–researcher gRPC certificates** configured in the
node's certificate settings or through `/api/certificates` endpoints. Configuring
one connection does not enable TLS for the other. See
[mutual TLS](../deployment/mutual-tls.md) for node–researcher authentication.

## Development and rebuilding assets

`--development` selects Flask's development server instead of Gunicorn.
`--debug` enables Flask debug mode and debug-level application logging; with
`--development`, it also enables the debugger and backend reloader.
Neither option is required for normal use.

For a source checkout, rebuild changed frontend sources before serving them:

```sh
fedbiomed-gui --path /path/to/node --recreate
# Equivalent:
fedbiomed node --path /path/to/node gui start --recreate
```

This requires Node.js, Yarn and frontend sources. It builds once; it does not
watch frontend changes. For live frontend development, see the
[development guide](../../developer/development-environment.md).

## Configuration file

Apart from `fedbiomed` command, some options can be configured through GUI configuration file and used without specifying each time the node is started. This file is located in node component directory, `/path/to/node-component/etc/config_gui.ini`.


### Server Configuration

The standalone launchers and core wrappers use `--host`, `--port` and
`--data-folder` to select these settings. Their defaults take precedence over
the following legacy server settings. For direct WSGI deployment, `DATA_PATH`
is used when the `DATA_PATH` environment variable is absent.

```ini
; --------------------------------------------------------------------------------------------
; Server configuration -----------------------------------------------------------------------
; --------------------------------------------------------------------------------------------
[server]

HOST = localhost
PORT = 8484
DATA_PATH = data
```

### Default Admin Configuration

When the Fed-BioMed GUI is started for the first time it will create a default admin with the credentials declared in the `[init_admin]` section of the configuration file. **By default, the email  will be `admin@fedbiomed.gui` and the password `admin`**. These settings seed the first administrator account only. Once the account exists, change its password through the User Panel or the API; editing this file does not update existing accounts.


```ini
;---------------------------------------------------------------------------------------------
; Initial admin credentials ------------------------------------------------------------------
; --------------------------------------------------------------------------------------------
[init_admin]

; --------------------------------------------------------------------------------------------
; - IMPORTANT!!! Please update initial admin credentials for production ----------------------
; --------------------------------------------------------------------------------------------
email = admin@fedbiomed.gui
password = admin
```

!!! note "Admin e-mail"
    Please modify admin e-mail address before starting the node GUI for the first time.
    Otherwise, it will create an admin with default
    e-mail address. If the admin is already created it can only be changed manually through database file.

!!! note "e-mail addresses"
    Currently, e-mail addresses are only used a login name by Fed-BioMed GUI. This is neither a user
    identity existing in the whole Fed-BioMed instance, nor used to send e-mails to the GUI user.

## Certificates and the researcher connection

The **Configuration** tab of the Node Management page holds what the node needs for
[mutual authentication](../deployment/mutual-tls.md), grouped under
*Connection & certificates*: its own certificate to send to the researcher, the
researcher certificate it registers and pins, and the `[authentication]` setting.
The **Connection & Diagnostics** tab reads the state of the connection to the
researcher as the node last observed it. Both are restricted to administrators, and
the certificate actions are the ones `fedbiomed node certificate` offers on the
command line.


## Upgrading from the combined package

The Python distributions are now separate: `fedbiomed` provides core,
`fedbiomed-node-api` provides HTTP services, and `fedbiomed-gui` adds the frontend.
The extras select matching package versions:

```sh
pip install --upgrade "fedbiomed[gui]"
pip check
```

For an API-only installation, use `fedbiomed[node-api]` instead. Update the
packages together; exact sibling pins reject incompatible versions.
The existing `fedbiomed node ... gui start` command remains supported.

The package split does not change existing node configuration files, dataset
registrations or user databases. Stop the services before updating and restart
with the same `--path` and data folder. Existing `etc/config_gui.ini` and
`var/gui_db_<node-id>.json` remain in use; no database migration is needed for
this split. Existing passwords are preserved.

For startup problems, see [API and GUI troubleshooting](../../support/troubleshooting.md#node-api-and-gui).
