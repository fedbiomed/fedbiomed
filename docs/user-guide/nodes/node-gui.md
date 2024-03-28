# Node GUI

Fed-BioMed provides a node user interface which currently allows node users to manage datasets and training plans easily.
The GUI can be used as an alternative to command line interface (CLI).  Since the implementation of GUI is still in beta
state it is only available for local access.

## Installing the Node GUI Environment

The conda environment `fedbiomed-gui` should be installed on the machine where the node GUI will be running.
Node and Node GUI should be running on same host/machine. This will allow GUI to access node's database and files.

!!! note "Technologies"
    The back-end of GUI is developed using Flask and front-end with ReactJS.

The following command will install GUI conda environment including `node.js` for `ReactJS` and necessary python libraries
for back-end APIs.

Please run the following command to install GUI conda environment.

```shell
$ ${FEDBIOMED_DIR}/scripts/configure_conda gui
```

## Starting Node GUI

The option `gui` of the script `fedbiomed_run` is configured for starting Node GUI. You can run
`${FEDBIOMED_DIR}/scripts/configure_conda gui --help` for usage and the description of the options, or you can follow
the sections for more detailed information.

!!! info "Attention!"
    By default `fedbiomed_run node gui [OPTIONS]` starts Flask server accepting access only from
    `localhost`. It is not safe to open access from remote host machine since it is not a secured
    web server yet. We highly recommend to use `localhost` through SSH Tunnel for remote access.


### Options to Start The GUI

Node GUI can be started through the script `fedbiomed_run` with various settings such as ip, port, folder where data files
are stored or the configuration that specifies the node that the GUI will run for.

The following command is the basic command to start Node GUI with default settings. This command assumes that the
data files are stored in `${FEDBIOMED_DIR}/data` and the default node config is `config_node.ini` which is stored in
`${FEDBIOMED_DIR}/etc`.

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui start`

After running this command the GUI will start listening on `localhost` on port `8484`.
You can access the GUI through
browser `http://localhost:8484`. This page will redirect you to the login page. The credentials and possible
configurations for log-in are explained in the [default admin configuration](#default-admin-configuration).

!!! warning "Important"
        If you are starting Fed-BioMed GUI with default settings please make sure that the `config_node.ini` is existing
        in `${FEDBIOMED_DIR}/etc`. Otherwise, starting operation will fail since it won't be able to find the node
        configuration. If you don't have `config_node.ini` yet please start the node before starting the GUI.


#### Using Different Port and Host

Custom ports and host IP address can be specified as long as the port in the specified IP isn't already in use.

```shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --port <port> --host <ip-address|localhost> start
```


#### Specifying Data Folder

You might want to store your data files in a different folder. In such cases you can use the option `--data-folder` to
specify which folder is used that includes data files.

````
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --data-folder <path/to/data/folder> start
````

!!! info "Uploading data files through Fed-BioMed is not allowed."
    Fed-BioMed assumes that the datasets or the datafiles that will be deployed in the node are already present in
    the data folder that is specified. Fed-BioMed Node GUI will help you to use these stored datasets in node.

#### Specifying Node Configuration

It is possible to specify the node that the user interface will be used for through the option `config`.

````
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --config <config-name>.ini start
````

Thanks to this option it is possible to start multiple GUI for multiple nodes on the same machine as long as the ports are different.


```shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --port 5001 --config config-node-1.ini start
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --port 5002 --config config-node-2.ini start
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --port 5003 --config config-name-3.ini start
```

If it is desired they can share the same data folder.


## Configuration file

Apart from `fedbiomed_run` command, some options can be configured through GUI configuration file and used
without specifying each time the node is started. This file is located in `${FEDBIOMED_DIR}/gui/config_gui.ini`.


### Server Configuration

You can modify `HOST`, `IP` and `DATA_PATH` (equivalent of `--data-folder`) in the server section of the configuration.

```ini
; --------------------------------------------------------------------------------------------
; Server configuration -----------------------------------------------------------------------
; --------------------------------------------------------------------------------------------
[server]

HOST = localhost
PORT = 8484
DATA_PATH = /data
```

### Default Admin Configuration

When the Fed-BioMed GUI is started for the first time it will create a default admin with the credentials declared
in the `[init_admin]` section of the configuration file. **By default, the email  will be `admin@fedbiomed.gui`
and the password `admin`**. You can modify the password either in configuration file or in GUI through User Panel
but the e-mail can only be modified from the configuration file.


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

## Production Mode

By default `fedbiomed_run node gui` launches Node GUI in development mode that uses Flask web server.  However,
using Flask server is not recommended for the production. Therefore, Node GUI has been configured to run on
[Gunicorn](https://gunicorn.org/) **application server** when the production mode is activated. Please type the following command to activate
production mode.

```shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --production start
```

It is also possible to activate production mode by setting  env variable `GUI_PRODUCTION=1` or `GUI_PRODUCTION=True`

```shell
$ GUI_PRODUCTION=1 ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --production start
```

!!! note "Please use a web server"
    [Gunicorn](https://gunicorn.org/) is an application server, and it is strongly recommended by [Gunicorn](https://gunicorn.org/)
    to use proxy web server such as [Nginx](https://www.nginx.com/) to  forward requests to Gunicorn using reverse proxy.

### Setting an SSL Certificate

[Gunicorn](https://gunicorn.org/) allows setting SSL certificate on application server layer. Please use following command to set SSL
certificate for the application server.

```shell
$ GUI_PRODUCTION=1 ${FEDBIOMED_DIR}/scripts/fedbiomed_run node gui --key-file <path-to-key-file> --cert-file <path-to-cert-file> start
```

SSL certificate can also be set through proxy server (e.g. [Nginx](https://www.nginx.com/)) instead of application server.

!!! note "Nginx proxy server for GUI is provided in VPN/containers deployment mode"
    Fed-BioMed provides a ready-to-deploy Node GUI container in VPN/containers deployment mode, that is configured to use [Nginx](https://www.nginx.com/) as a
    proxy server and [Gunicorn](https://gunicorn.org/) as an application server. This also allows for setting custom SSL certificates.
    Please refer to the  [VPN deployment](../deployment/deployment-vpn.md) documentation.
