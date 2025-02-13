# Node GUI

Fed-BioMed offers a node user interface that allows users to manage datasets and training plans with ease. This graphical user interface (GUI) serves as an alternative to the command-line interface (CLI). However, since the GUI is currently in its beta stage, it is only accessible locally.

## Installing the Node GUI Dependencies

Node GUI dependencies are not installed by default with the standard Fed-BioMed installation. They are provided as an extra module in the Fed-BioMed package. To install them, use `pip` with the `gui` option specified, as shown below:

```
pip install fedbiomed[gui]
```


## Starting Node GUI

The option `gui` of `fedbiomed` command is configured for starting Node GUI.

!!! info "Attention!"
    By default `fedbiomed node gui start [OPTIONS]` starts Flask server accepting access only from
    `localhost`. It is not safe to open access from remote host machine since it is not a secured
    web server yet. We highly recommend to use `localhost` through SSH Tunnel for remote access.


### Options to Start The GUI


The Node GUI in Fed-BioMed can be launched using the `fedbiomed` command with various customizable settings, such as the IP address, port, folder for storing data files, and the configuration specifying the Node the GUI will manage.

The following command demonstrates how to start the Node GUI with its default settings. This command uses the default Fed-BioMed Node component directory, which corresponds to the directory where the command is executed. If no Node component exists in that directory, a new one will be automatically created.

By default, the Node GUI assumes that data files are stored in the `data` directory within the Node component folder. For example, if the command is executed in `/path/to/workdir`, the Node component will be instantiated in `/path/to/workdir/fbm-node/`, with the default data directory located at `/path/to/workdir/fbm-node/data`.

```
fedbiomed node gui start
```

After running this command the GUI will start listening on `localhost` on port `8484`. You can access the GUI through browser `http://localhost:8484`. This page will redirect you to the login page. The credentials and possible configurations for log-in are explained in the [default admin configuration](#default-admin-configuration).

#### Using different port and host

Custom ports and host IP address can be specified as long as the port in the specified IP isn't already in use.

```shell
$ fedbiomed node gui start --port <port> --host <ip-address|localhost>
```


#### Specifying data folder

You might want to store your data files in a different folder. In such cases you can use the option `--data-folder` to specify which folder is used that includes data files.

````
$ fedbiomed node gui start --data-folder <path/to/data/folder>
````

!!! info "Uploading data files through Fed-BioMed is not allowed."
    Fed-BioMed assumes that the datasets or the datafiles that will be deployed in the node are already present in the data folder that is specified. Fed-BioMed Node GUI will help you to use these stored datasets in node.

#### Specifying specific node component whose GUI will be launched

It is possible to specify the node that the user interface will be used for through the option `--path` or `-p`.

```
$ fedbiomed node --path <path/to/component/directory> gui start
````

Thanks to this option it is possible to start multiple GUI for multiple nodes on the same machine as long as the ports are different.


```shell
$ fedbiomed node --path ./my-first-node gui start --port 5001
$ fedbiomed node --path ./my-second-node gui start --port 5002
$ fedbiomed node --path ./my-third-node gui start --port 5003
```

If it is desired they can share the same data folder.


## Configuration file

Apart from `fedbiomed` command, some options can be configured through GUI configuration file and used without specifying each time the node is started. This file is located in node component directory, `/path/to/node-component/etc/config_gui.ini`.


### Server Configuration

You can modify `HOST`, `IP` and `DATA_PATH` (equivalent of `--data-folder`) in the server section of the configuration.

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

When the Fed-BioMed GUI is started for the first time it will create a default admin with the credentials declared in the `[init_admin]` section of the configuration file. **By default, the email  will be `admin@fedbiomed.gui` and the password `admin`**. You can modify the password either in configuration file or in GUI through User Panel but the e-mail can only be modified from the configuration file.


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

By default, `fedbiomed node gui` launches the Node GUI in production mode, utilizing [Gunicorn](https://gunicorn.org/) as the application server. For debugging and development purposes, you can launch the GUI using the `--development` flag.

```shell
$ fedbiomed node gui start --development
```


!!! note "Please use a web server"
    [Gunicorn](https://gunicorn.org/) is an application server, and it is strongly recommended by [Gunicorn](https://gunicorn.org/)
    to use proxy web server such as [Nginx](https://www.nginx.com/) to  forward requests to Gunicorn using reverse proxy.

### Setting an SSL Certificate

[Gunicorn](https://gunicorn.org/) allows setting SSL certificate on application server layer. Please use following command to set SSL
certificate for the application server.

```shell
$ fedbiomed node gui start --key-file <path-to-key-file> --cert-file <path-to-cert-file>
```

SSL certificate can also be set through proxy server (e.g. [Nginx](https://www.nginx.com/)) instead of application server.

!!! note "Nginx proxy server for GUI is provided in VPN/containers deployment mode"
    Fed-BioMed provides a ready-to-deploy Node GUI container in VPN/containers deployment mode, that is configured to use [Nginx](https://www.nginx.com/) as a
    proxy server and [Gunicorn](https://gunicorn.org/) as an application server. This also allows for setting custom SSL certificates.
    Please refer to the  [VPN deployment](../deployment/deployment-vpn.md) documentation.
