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


## User Accounts

The GUI enforces authentication for all pages except the login and registration screens. There are two roles:

| Role | Description |
|---|---|
| **Admin** | Full access including user management, registration approval, and security logs |
| **User** | Access to datasets, training plans, node management, and file repository |

### Default Admin Account

On first startup, the GUI creates an admin account automatically using the credentials from the `[init_admin]` section of `config_gui.ini`. The defaults are:

- **Email:** `admin@fedbiomed.gui`
- **Password:** `admin`

!!! warning "Change the default credentials"
    Update the email and password in `config_gui.ini` **before** the first startup. Once the admin is created, the email can only be changed directly in the GUI database file (`<node-root>/var/gui_db_<NODE_ID>.json`).

### Password Requirements

All passwords (for both self-registration and admin-created accounts) must satisfy:

- At least 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit


### Creating a User Account (Admin)

Admins can create accounts that are immediately active — no approval step required.

1. Log in as an admin.
2. Navigate to **User Account → User Management**.
3. Click **Create new account**.
4. Fill in name, surname, email, and password, then submit.

The new user can log in straight away.

Additional actions available from the same page:

| Action | Description |
|---|---|
| **Reset password** | Generates a random 12-character password and displays it once |
| **Change role** | Promote a user to Admin or demote to User |
| **Delete user** | Permanently removes the account (admins cannot delete their own account) |


## Registration

Users who do not have an account can request one through the self-registration form at `/register/`. The request is held in a pending state until an admin approves or rejects it.

### Submitting a Registration Request

1. Open the GUI in a browser and click **Register** on the login page (or navigate to `/register/`).
2. Fill in name, surname, email, and password.
3. Submit the form.

The server creates a pending request. The user **cannot log in** until an admin approves the request. A confirmation message is displayed: *"A request has been sent to the administrator for account creation."*

### Approving or Rejecting Requests (Admin)

1. Log in as an admin.
2. Navigate to **User Account → Account Requests**.
3. The table lists all pending and rejected requests with the requester's name, email, and submission date.

| Button | Effect |
|---|---|
| **Approve** | Moves the request to the Users table — the user can log in immediately with the password they set during registration |
| **Reject** | Marks the request as Rejected — the record remains visible and can still be approved later |


## Pages Overview

### Home

**Route:** `/`

The landing page after login. Displays quick-link cards for the main sections: Documentation, List Files, Dataset Management, Load Datasets, and Configuration.

---

### List Data Files (Repository)

**Route:** `/repository/`

A file browser for the node's data directory. Supports column view and list view. Use this page to navigate the folder structure of the configured data path and identify files before registering them as datasets.

---

### Training Plans

**Route:** `/training-plans/`

Lists all training plans (models) that have been submitted to the node. Each entry shows the plan name and its current approval status: **Pending**, **Approved**, or **Rejected**.

Click on a training plan to open its detail view (`/training-plans/preview/:id`), where you can:

- Read the full model source code
- **Approve** — allow the node to execute this plan in federated rounds
- **Reject** — prevent execution
- **Delete** — remove the plan from the node

Training plan approval is part of the node's security model. Only approved plans can be used in experiments. See [Training Plan Security Manager](training-plan-security-manager.md) for more details.

---

### Datasets

**Route:** `/datasets/`

Lists all datasets currently registered on the node. Supports keyword search and dataset removal. An **Add MNIST Dataset** shortcut is available for quick testing.

Click on a dataset to open its preview (`/datasets/preview/:id`), which shows:

- Metadata: name, description, tags, data type, shape
- A preview of the data (first rows of a CSV, or folder listing for image/medical datasets)

#### Adding a Dataset

**Route:** `/datasets/add-dataset/`

A multi-step wizard for registering a new dataset. The first step asks you to choose a dataset type:

| Type | Description |
|---|---|
| **CSV** | Tabular data from a `.csv` file |
| **Images** | A folder of image files |
| **Medical Folder** | A structured folder following the Medical Folder Dataset convention (subjects × modalities) |

For **Medical Folder datasets**, the wizard walks through additional steps: validating the root directory, selecting a reference CSV, mapping subject modalities, and optionally defining a DataLoadingPlan.

---

### Node Management

**Route:** `/node-management/`

Controls the Fed-BioMed node process and provides access to application logs. Two tabs:

**Process Details**

- Start, stop, or restart the node process directly from the browser
- View current status (running / stopped / stopping), PID, uptime, and GPU configuration
- Set GPU options (CUDA device index, disable GPU) before starting

**Application Logs**

- Browse log files produced by the node
- Paginated, cursor-based backwards log viewer
- Download raw log files

---

### Node Configuration

**Route:** `/configuration/`

Read-only view of the node's configuration. Displays:

- Node ID and name
- Database path
- Training plan approval mode (auto-approve or manual review required)
- Whether default training plans are allowed
- Hashing algorithm in use

---

### User Account

**Route:** `/user-account/`

Personal and administrative account settings, organized as tabs:

| Tab | Access | Description |
|---|---|---|
| **My Info** | All users | Displays your name, email, and role |
| **Change Password** | All users | Update your own password (current password required) |
| **User Management** | Admin only | Create, delete, reset password, or change role for any user |
| **Account Requests** | Admin only | Review, approve, or reject pending registration requests |
| **Security Logs** | Admin only | Filterable audit log of security-relevant events (login attempts, training plan actions, etc.). Supports filtering by operation type, status, researcher ID, date range, and keyword search |
