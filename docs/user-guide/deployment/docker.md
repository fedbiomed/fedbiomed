# Deploying Fed-BioMed Using Docker Image

## Introduction

Fed-BioMed provides a set of Docker images designed to simplify the deployment, configuration, and testing of its core components. These images are intended to help users get started quickly, without the need to manually install dependencies or configure environments for each component. Whether you're a researcher experimenting with federated learning workflows or a developer integrating Fed-BioMed into a larger system, these Docker images serve as a ready-to-use components of Fed-BioMed.

The Docker images not only support the standard use cases but are also designed to be easily extendable. You can customize or build on top of the base images to suit their specific requirements, such as adding wrapping Fed-BioMed within your application, integrating new data sources, or generally adapting the behavior of components like the node or researcher to align with your infrastructure.

This documentation will guide you through the available Docker images for each Fed-BioMed component, explain how to launch and manage them, and provide best practices for extending and customizing the containers to fit your needs.
 
## Docker Images and Containers 

Fed-BioMed Docker images are published for each released version of Fed-BioMed.
You can visit [Docker Hub](https://hub.docker.com/u/fedbiomed) to see the available images published by the Fed-BioMed team.

!!! warning "Component Versions"
    Although Fed-BioMed components are generally backward compatible, it is recommended to use the same version across all component images to ensure full compatibility and stability.

---

Fed-BioMed provides two main images as `fedbiomed/node` and `fedbiomed/researcher` to run minimal federated learning infrastructure.

Here is the list of docker images:

- **fedbiomed/base**: Base Fed-BioMed image that comes with vanilla Fed-BioMed installation that can be extended later.
- **fedbiomed/node**: Image that comes with Fed-BioMed node dependencies installed.
- **fedbiomed/researcher**: Image that comes with Fed-BioMed researcher utilities and dependencies installed.

### Basic commands to quickly launch Fed-BioMed instances 

The following commands will pull the requested version and start containers with default configurations. However, depending on your OS and network configuration, you may need to configure networking to establish connections between the federation server and Fed-BioMed nodes.

#### Basic Setup (No Network Configuration)

```bash
docker run --name fbm-node-1 fedbiomed/node:<version-tag>
```

```bash
docker run --name fbm-node-2 fedbiomed/node:<version-tag>
```

```bash
docker run --name fbm-researcher fedbiomed/researcher:<version-tag>
```

#### Network-Configured Setup

For proper communication between components, use a Docker network:

**Create the network:**
```bash
docker network create fedbiomed-net
```

**Start Node 1:**
```bash
docker run --network fedbiomed-net \
  --name fbm-node-1 \
  -e FBM_RESEARCHER_IP=fbm-researcher \
  -e FBM_RESEARCHER_PORT=50051 \
  fedbiomed/node:<version-tag>
```

**Start Node 2:**
```bash
docker run --network fedbiomed-net \
  --name fbm-node-2 \
  -e FBM_RESEARCHER_IP=fbm-researcher \
  -e FBM_RESEARCHER_PORT=50051 \
  fedbiomed/node:<version-tag>
```

**Start Researcher:**
```bash
docker run --network fedbiomed-net \
  --name fbm-researcher \
  -p 8888:8888 \
  -e FBM_SERVER_HOST=fbm-researcher \
  fedbiomed/researcher:<version-tag>
```

Replace `<version-tag>` with a version available on Docker Hub to launch a local federated learning network. The setup comes ready to use with the MNIST dataset. After launching the commands, navigate to `localhost:8888` to execute a basic Fed-BioMed notebook using MNIST.


### Configuration

Fed-BioMed docker images keeps Fed-BioMed node and researcher configuration and folders respectivelty under `/fbm-node` and `/fbm-researcher`. It is recommended to mount this directories in the host machine and configure manually via `config.ini`. 

Docker provides several options for storing container data on the host machine. Bind mounting and Docker Volumes are the most common approaches.  Docker named volumes are recommended over bind-mounted directories for several reasons: they perform better on non-Linux hosts, are portable and easy to backup, remain isolated from the host, facilitate data sharing between containers, and are auto-populated from the image without overwriting existing data. However, bind mounts can be more convenient for testing and development purposes.

Component configuration (node, researcher or gui) can be manipulated at runtime by assigning environment variables using the `--env` or `-e` flag. These variables will also define the initial configuration values during the first run of the container.

After the Docker container is started for the first time, the node configuration is created inside the container under `/fbm-node/etc` or `/fbm-researcher` for researcher instance. The config folder hiearhcy is same for all the Fed-BioMed components. To persist this configuration across runs or to edit it manually, mount a local directory or docker volume using the `-v` option:

If it is a Docker volume:

```bash
-v fedbiomed-node-volume:/fbm-node
```

To modify the configuration, edit the file locatedin the `fedbiomed-node-volume` volume. First, check the volume directory:

```bash
docker volume inspect fedbiomed-node-volume
```

```json
[
    {
        "CreatedAt": "xxx",
        "Driver": "local",
        "Labels": null,
        "Mountpoint": "/var/lib/docker/volumes/fedbiomed-node-volume/_data",
        "Name": "fedbiomed-node-volume",
        "Options": null,
        "Scope": "local"
    }
]
```

After the configuration is changed, the Fed-BioMed container needs to be restarted:

```bash
docker restart <name-of-container>
```


#### Configuration Behavior Overview:

- **First run with environment variables**:
    If the container is run for the first time with environment variables, a new node configuration will be created using those values.

- **Using an existing configuration directory**:
    If a previously created node directory is mounted, the container will use the configuration found there.

- **Overriding existing configuration at runtime**:
    If a static configuration already exists but environment variables are still provided, those variables will override the config **only at runtime**, without modifying the actual `config.ini` file.

### Bind-mounted directory

A bind mount maps a directory from your host machine directly into the container. This makes it easy to inspect or edit configuration files without entering the container.

To mount a local directory, add the `-v` flag with an absolute path:

```bash
# For a node container
docker run --network fedbiomed-net \
  --name fbm-node-1 \
  -v <absolute-path-to-host-fbm-node>:/fbm-node \
  -e FBM_RESEARCHER_IP=fbm-researcher \
  -e FBM_RESEARCHER_PORT=50051 \
  fedbiomed/node:<version-tag>

# For a researcher container
docker run --network fedbiomed-net \
  --name fbm-researcher \
  -v <absolute-path-to-host-fbm-researcher>:/fbm-researcher \
  -p 8888:8888 \
  -e FBM_SERVER_HOST=fbm-researcher \
  fedbiomed/researcher:<version-tag>
```

!!! warning "Permission issues on Linux"

    On Linux, Docker creates the mounted directory on the host as `root` if it does not already exist. The container process runs as `fedbiomed` (UID/GID `2101:2101` by default) and cannot write to a root-owned directory, causing startup failures.

    **Option 1 — Use Docker named volumes (recommended)**

    Prefer Docker named volumes (e.g. `-v fedbiomed-node-volume:/fbm-node`) over bind mounts. Docker manages ownership automatically and avoids this problem entirely.

    **Option 2 — Use `CONTAINER_UID` / `CONTAINER_GID`**

    Start the container as `root` and pass your host user's UID/GID. The entrypoint remaps the internal `fedbiomed` user to match, so files created in the mounted directory are owned by you. See [Launching docker container with a different user](#launching-docker-container-with-a-different-user).

    **Option 3 — Pre-create the directory with group write access**

    Create the directory before running Docker and grant write permission to a `fedbiomed` group with GID `2101` (see also [Default Container User](#default-container-user)):

    ```bash
    sudo groupadd -g 2101 fedbiomed
    mkdir -p /mounted/fbm-node
    sudo chown -R :fedbiomed /mounted/fbm-node   # assign the group
    sudo chmod -R g+w /mounted/fbm-node           # allow group writes
    sudo chmod g+s /mounted/fbm-node              # new files inherit the group
    sudo usermod -aG fedbiomed $USER              # add yourself to the group
    # log out and back in for the group change to take effect
    ```

    Then start the container normally:

    ```bash
    docker run --network fedbiomed-net \
      --name fbm-node-1 \
      -v /mounted/fbm-node:/fbm-node \
      -e FBM_RESEARCHER_IP=fbm-researcher \
      -e FBM_RESEARCHER_PORT=50051 \
      fedbiomed/node:<version-tag>
    ```

### Default Container User

All Fed-BioMed images run component processes under a non-root user named `fedbiomed`. This user and its group are baked into the image with fixed IDs:

| Identity | Name        | ID     |
|----------|-------------|--------|
| User     | `fedbiomed` | `2101` |
| Group    | `fedbiomed` | `2101` |

By default, the container starts directly as the `fedbiomed` user — no root involvement. If you explicitly start the container as `root` (using `--user root`) to fix volume ownership, the entrypoint script (`/entrypoint-base.sh`) performs the ownership changes first and then drops privileges to `fedbiomed` before launching the component process. This means:

- All Fed-BioMed processes run without root privileges.
- Files written inside the container (configuration, data, logs) are owned by `fedbiomed:fedbiomed` (UID/GID `2101:2101`).
- You do **not** need to pass `--user fedbiomed` to `docker run` in normal use — the image's default user is already `fedbiomed`.

When opening a shell inside a running container with `docker exec`, pass `-u fedbiomed` to match the user running the component:

```bash
docker exec -it -u fedbiomed my-node bash
```

!!! note "Remapping to your host user"
    Files created by the container will be owned by UID `2101` on the host, which may not match your own account. If this causes permission issues with bind-mounted directories, see [Launching docker container with a different user](#launching-docker-container-with-a-different-user) to remap the `fedbiomed` user to your host UID/GID at runtime.

---

### Launching docker container with a different user

By default, the Fed-BioMed Node component is launched using a predefined user inside the Docker container. However, you can specify a different user at runtime to avoid permission issues (especially using bind-mounting directories) when working with files on your local machine. Docker containers keep the default user `fedbiomed` but change the user ID to match the requested user ID from the host to solve the permission issues. The environment variables that need to be set are `CONTAINER_UID` and `CONTAINER_GID`.

Here's how to run the container with custom user settings:

```bash
docker run -it \
    --user root \
    --name my-node \
    -v <path-to-local-fbm-node>:/fbm-node \
    -v fbm-node-volume:/fbm-node
    -e CONTAINER_UID=<user-id> \
    -e CONTAINER_GID=<group-id> \
    fedbiomed/node:latest
```

Please, replace `<user-id>`, and `<group-id>` with the corresponding values from your host system.

!!! note "**Ownership** transfer can take some time"
    Depending on the file size, ownership transfer from the Docker Fed-BioMed default user to the host user may take some time.


## Fed-BioMed Component Type Specifications 

### Nodes  

The following command pulls the Fed-BioMed Node image and runs it. Running this image will automatically start a Fed-BioMed Node component.

!!! note "Running in background"
    Please add the `-d` option to run the Docker container in the background.

```bash
docker run -it  \
    --name my-node \
    --network host \ 
    -v fbm-node-volume:/fbm-node
    -e FBM_SECURITY_SECURE_AGGREGATION=True \
    -e FBM_RESEARCHER_PORT=50051 \
    -e FBM_RESEARCHER_IP=localhost \
    fedbiomed/node:latest
```

For testing purposes, it's fine to use the default configuration. However, for deployment, you may want to review the available configuration options to ensure the Node is properly set up. Component configuration can be managed using environment variables. All environment variables listed in the [Node configuration guide](../nodes/configuring-nodes.md) can be passed to the container using Docker’s `--env` option. 

It is recommended to update the configuration directly in the file located at `<absolute-path-to-mounted-fbm-node>/etc/config.ini`. This approach provides a more stable and persistent setup. Please see the section [Configuration](#configuration) for more details.

### Managing the Node Process

The Fed-BioMed node runs as a background process inside the container, managed by Fed-BioMed node process manager.

To interact with the node process, open a shell inside the running container:

```bash
docker exec -it -u fedbiomed <my-node-name> bash
```

Once inside, use the Fed-BioMed CLI to inspect and control the node:

```bash
# Check whether the node process is running
fedbiomed node -p /fbm-node status

# Stop the node process
fedbiomed node -p /fbm-node stop

# Restart the node process
fedbiomed node -p /fbm-node restart

# Start the node if it has been stopped
fedbiomed node -p /fbm-node start
```

!!! note "Restart after configuration changes"
    After modifying the node configuration — whether via environment variables or directly in `config.ini` — run `fedbiomed node restart` inside the container for the changes to take effect. There is no need to restart the Docker container itself.

!!! tip "Node restart vs. container restart"
    - `fedbiomed node restart` — restarts only the **node process** inside the running container. The container and supervisord keep running.
    - `docker restart my-node` — restarts the **entire container**, which re-runs the entrypoint script, re-applies environment variables, and reinitializes the node from scratch.

---


#### Adding Datasets to Nodes Launched in Containers

There are two ways to add datasets to Docker containers running Fed-BioMed nodes:

1. **Using the CLI** inside the container console.
2. **Using the Node GUI** to add and manage datasets.

For GUI instructions, please see the [Node GUI](#node-gui) section.

This section focuses on adding datasets using the Fed-BioMed CLI by accessing the container's terminal.


##### 1. Mounting a Dataset Directory

To add dataset files to the container, it is **mandatory** to mount a local directory to `/fbm-node` inside the container:

```bash
-v <path-to-host-fbm-node>:/fbm-node
```

Within `<path-to-host-fbm-node>`, a `data` directory will be created. You can copy your dataset files into this directory. These files will then be visible and accessible inside the container.


##### 2. Accessing the Container Console

Once your dataset files are placed in the `data` directory, you can enter the container's terminal to use the CLI and register the dataset:

```bash
docker exec -it -u fedbiomed my-node bash
```

!!! warning "Test" 
    Make sure to replace `fedbiomed` with the appropriate container user if it's different.


##### 3. Registering the Dataset via CLI

Once inside the container, use the Fed-BioMed CLI to add the dataset:

```bash
$ fedbiomed node -p /fbm-node dataset add
```

For more details, see the full guide on [deploying datasets](../nodes/deploying-datasets.md).

---

!!! note "MNIST Default Dataset"

    To simplify testing, the MNIST dataset is automatically deployed when the container is launched.  
    You can disable this by adding the following environment variable in your Docker run command:

    ```bash
        docker ....
        ...
        -e MNIST_OFF=True
        ...
        fedbiomed/node
    ```

---

!!! note "GPU Support"
    Docker containers can utilize GPUs as long as the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is properly installed and configured. Once the toolkit is set up, you can enable GPU support by using the `--gpus all` flag with the `docker run` command, or by using the `device_requests` section in a Docker [Compose](https://docs.docker.com/compose/how-tos/gpu-support/) file.

    The `fedbiomed/node` image installs PyTorch on Linux with CUDA support bundled — no NVIDIA base image is required. The NVIDIA Container Toolkit exposes the GPU device to the container, and PyTorch uses its bundled CUDA runtime to communicate with it.

!!! warning "CUDA Version Compatibility"
    PyTorch bundles a specific CUDA version in its wheel (e.g. `cu126`). The NVIDIA driver on your **host machine** must support that CUDA version or newer. If the driver is too old, PyTorch will not detect the GPU even when `--gpus all` is set.

    To check the maximum CUDA version your driver supports, run on the host:

    ```bash
    nvidia-smi
    ```

    The output shows the **CUDA Version** in the top-right corner. This must be greater than or equal to the CUDA version bundled with the installed PyTorch wheel. If it is not, update your NVIDIA driver before launching the container.

---

### Node GUI

The Node GUI is bundled directly into the `fedbiomed/node` image — no separate image is required. When the node container starts, the GUI server is launched automatically alongside the node process and is accessible on port **8484**.

To expose the GUI to your host machine, forward port `8484` when starting the container:

```bash
docker run -it \
    --name my-node \
    -v <path-to-host-fbm-node>:/fbm-node \
    -p 8484:8484 \
    fedbiomed/node:latest
```

Once running, open your browser and navigate to `http://localhost:8484`.

If you are running multiple node containers, map each container's port `8484` to a different host port:

```bash
# Node 1 GUI on host port 8484
docker run -it --name node1 \ 
  -v ./node1:/fbm-node \
  # or -v fedbiomed-node-wolume-node-1:/fbm-node \
  -p 8484:8484 \
  fedbiomed/node:latest

# Node 2 GUI on host port 8485
docker run -it --name node2 \
  -v ./node2:/fbm-node \
  # or -v fedbiomed-node-wolume-node-2:/fbm-node \
  -p 8485:8484 \
  fedbiomed/node:latest
```

!!! note "GUI starts with the node"
    The GUI server is managed by supervisord inside the container and starts automatically with the node. There is no separate start command needed.

---

## Researcher

As it is in Node image researcher node work in a same way to lancuhs and configure. Environment variables can be used to manipulate researcher configuration or researcher configuration file located in the mounted directory can changed. 

The following command will launch a researcher component with a basic configuration. It is recommended to mount the volume for `/fbm-researcher` directory in host that generated and keeps researcher component configurations, folders and files. Researcher component exposes a port that allows other component. Researcher component also exposes port tensorbard application to display feedback scalar values collected from participating node during the training.  

```bash
docker run -it -d \ 
    -v <path-fedbiomed-researcher>:/fbm-researcher \
    # or -v fedbiomed-researcher-volume:/fbm-researcher \
    -p 50051:50051 \
    fedbiomed/researcher:latest
```

### Configuring Networking for the Fed-BioMed Researcher Docker Container

There are several ways to configure Docker container networking depending on the deployment scenario. Therefore, it is up to the user to choose the network settings that best fit their needs. However, Fed-BioMed provides two important environment variables for configuring the Researcher container:


* `FBM_SERVER_HOST`
* `FBM_SERVER_PORT`

These correspond to the `host` and `port` entries in the `server` section of the Researcher’s configuration file. They **must be set correctly** to ensure that Fed-BioMed Node containers can connect to the Researcher.

By default, the Docker image sets `FBM_SERVER_HOST=0.0.0.0`, which allows the Researcher to **accept connections from any interface**. This is useful when Nodes run on **different machines** or in **separate Docker networks**.

Example: Using `127.0.0.1`

In the following example, the Researcher is configured to bind to `127.0.0.1` (localhost):

```bash
docker run -it -d \
    -e FBM_SERVER_HOST=127.0.0.1 \
    -v <path-fedbiomed-researcher>:/fbm-researcher \
    # or -v fedbiomed-node-wolume-node-2:/fbm-node \
    -p 50051:50051 \
    fedbiomed/researcher:latest
```

!!! warning "Limitation"
    With `FBM_SERVER_HOST=127.0.0.1`, the container listens only on its internal loopback interface. As a result, **other Node containers on the same host will not be able to connect**, unless they are attached to the **same Docker network**.


You can resolve this by attaching the containers to a user-defined Docker network:

```bash
# Create the network (only once)
docker network create fedbiomed-net

# Run the Researcher on that network
docker run -it -d \
    --network fedbiomed-net \
    --name fbm-researcher \
    -e FBM_SERVER_HOST=0.0.0.0 \
    -v <path-fedbiomed-researcher>:/fbm-researcher \
    fedbiomed/researcher:latest
```

Then launch the Node container using the same network:

```bash
docker run -it \
    --network fedbiomed-net \
    fedbiomed/node
```

This setup allows the Node container to connect to the Researcher via its container name (`fbm-researcher`), thanks to Docker's internal DNS.


It is up to the user to configure Docker networking according to their deployment scenario. For more information about Docker networking and available drivers, refer to the [Docker networking documentation](https://docs.docker.com/engine/network/drivers/).


## Use Docker Compose to Run and Manage Multiple Fed-BioMed Instances

Docker Compose lets you define and start all Fed-BioMed components with a single command, rather than running each container manually. This is the recommended approach for running a full federated learning setup locally.

The example below launches **one researcher** and **two nodes**, each with its own Node GUI.

### Step 1 — Create the project directory

Create a working directory for the Compose project:

```bash
mkdir fbm-setup
cd fbm-setup
```

Docker will create and manage the named volumes automatically on first run — no need to create directories manually.

### Step 2 — Create the `docker-compose.yml`

Create a file named `docker-compose.yml` inside `fbm-setup/` with the following content:

```yaml
services:
  researcher:
    image: fedbiomed/researcher:latest
    container_name: fbm-researcher
    environment:
      - FBM_SERVER_HOST=fbm-researcher
      - FBM_SERVER_PORT=50051
    volumes:
      - fbm-researcher:/fbm-researcher
    ports:
      - "50051:50051"   # gRPC server (nodes connect here)
      - "8888:8888"     # Jupyter notebook
      - "6007:6007"     # TensorBoard proxy
    networks:
      - fedbiomed-net

  node1:
    image: fedbiomed/node:latest
    container_name: fbm-node1
    environment:
      - FBM_RESEARCHER_IP=fbm-researcher
      - FBM_RESEARCHER_PORT=50051
      - FBM_SECURITY_TRAINING_PLAN_APPROVAL=False
    volumes:
      - fbm-node1:/fbm-node
    ports:
      - "8484:8484"     # Node GUI
    networks:
      - fedbiomed-net
    depends_on:
      - researcher

  node2:
    image: fedbiomed/node:latest
    container_name: fbm-node2
    environment:
      - FBM_RESEARCHER_IP=fbm-researcher
      - FBM_RESEARCHER_PORT=50051
      - FBM_SECURITY_TRAINING_PLAN_APPROVAL=False
    volumes:
      - fbm-node2:/fbm-node
    ports:
      - "8485:8484"     # Node GUI (mapped to 8485 to avoid conflict with node1)
    networks:
      - fedbiomed-net
    depends_on:
      - researcher

volumes:
  fbm-researcher:
  fbm-node1:
  fbm-node2:

networks:
  fedbiomed-net:
    driver: bridge
```

!!! note "Node GUI is built into the node image"
    Each node exposes its GUI directly on port `8484`. No separate GUI container is needed. Node 1 GUI is accessible at `http://localhost:8484` and Node 2 at `http://localhost:8485`.

!!! note "GPU Support"
    To enable GPU access for a node, add the following block under the node service. This requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to be installed on the host. See the [GPU compatibility warning](#) above for driver requirements.

    ```yaml
    node1:
      ...
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
    ```

### Step 3 — Start the services

From the `fbm-setup/` directory, run:

```bash
docker compose up -d
```

This starts all containers in detached mode. The named volumes (`fbm-researcher`, `fbm-node1`, `fbm-node2`) are created automatically and populated with the initial Fed-BioMed configuration on first run.

### Step 4 — Verify the setup

Check that all containers are running:

```bash
docker compose ps
```

You should see `fbm-researcher`, `fbm-node1`, and `fbm-node2` listed with status `Up`.

Open the Jupyter notebook to run experiments:

```
http://localhost:8888
```

Open the Node GUI to manage datasets:

```
http://localhost:8484   ← Node 1
http://localhost:8485   ← Node 2
```

### Managing the services

```bash
# View logs for all services
docker compose logs -f

# View logs for a specific service
docker compose logs -f node1

# Stop all services (data is preserved in the mounted folders)
docker compose down

# Restart a single service
docker compose restart node1
```

## Building Docker Images from Source

If desired, it is possible to build Fed-BioMed Docker images from source with custom modifications, such as changing the default user. This process is described in the [building Fed-BioMed Docker image documentation](../../developer/docker-images.md) for developers.

## Extending Docker Images

Fed-BioMed images are designed to be extended. Before customizing, it is important to understand how the default entrypoint chain works so you can choose the right approach for your use case.

### Understanding the Entrypoint Chain

Fed-BioMed images use a two-level entrypoint chain:

```
tini → /entrypoint-base.sh → /entrypoint.sh
```

- **`/entrypoint-base.sh`** (from `fedbiomed/base`) — runs when the container starts. If the container is launched as root, it handles UID/GID remapping (matching the internal `fedbiomed` user to `CONTAINER_UID`/`CONTAINER_GID`), fixes ownership of mounted directories, and drops privileges before handing off. It always calls `/entrypoint.sh` at the end.
- **`/entrypoint.sh`** (from each component image, e.g. `fedbiomed/node`) — runs as the `fedbiomed` user. It initializes the Fed-BioMed component, writes configuration, deploys default datasets, and starts the node or researcher process.

Depending on your needs, there are three ways to extend this chain.

---

### Scenario 1 — Custom process, UID/GID remapping preserved

**Use case:** You want the Fed-BioMed software available but you manage the process yourself (custom process manager, different startup logic). UID/GID remapping still works so mounted volumes are handled correctly.

**How:** Copy your own `/entrypoint.sh` into the image. Do **not** override `CMD`. `entrypoint-base.sh` remains the default entry point and will call your `/entrypoint.sh` after handling permissions.

```Dockerfile
FROM fedbiomed/node:<version-tag>

USER root
RUN pip install <your-package>

COPY my-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER fedbiomed
```

`my-entrypoint.sh`:

```bash
#!/bin/bash

# Your custom startup logic — the node does NOT start automatically here.
# You have access to the full fedbiomed CLI and all installed packages.
echo "Starting custom process..."
exec my-custom-process
```

The chain becomes:

```
tini → /entrypoint-base.sh (UID/GID remapping, privilege drop)
                          → /entrypoint.sh  ← your script
```

---

### Scenario 2 — Full override, software only

**Use case:** You only need Fed-BioMed installed as a library. You manage everything yourself and do not need UID/GID remapping or automatic component startup.

**How:** Override `CMD` with your own script. `entrypoint-base.sh` is bypassed entirely.

```Dockerfile
FROM fedbiomed/node:<version-tag>

USER root
RUN pip install <your-package>

COPY my-entrypoint.sh /my-entrypoint.sh
RUN chmod +x /my-entrypoint.sh

USER fedbiomed

CMD ["/my-entrypoint.sh"]
```

!!! warning
    This bypasses `entrypoint-base.sh` completely. UID/GID remapping (`CONTAINER_UID`/`CONTAINER_GID`) will not work and the node will not start automatically.

---

### Scenario 3 — Keep default behavior and add custom operations

**Use case:** You want the node to start exactly as it normally would, but you need to run additional operations before it starts (e.g. pre-loading a dataset, writing extra configuration, registering an external service).

**How:** Override `CMD` with a custom wrapper that does its work first, then calls `exec /entrypoint-base.sh` to hand off to the full chain.

```Dockerfile
FROM fedbiomed/node:<version-tag>

USER root
RUN pip install <your-package>

COPY my-entrypoint.sh /my-entrypoint.sh
RUN chmod +x /my-entrypoint.sh

USER fedbiomed

CMD ["/my-entrypoint.sh"]
```

`my-entrypoint.sh`:

```bash
#!/bin/bash

# --- Your custom logic runs here, before the node starts ---

# Example: register an environment-specific CA certificate
echo "Installing custom CA certificate..."
cp /run/secrets/org-ca.crt /usr/local/share/ca-certificates/org-ca.crt
update-ca-certificates

# Hand off to the full Fed-BioMed chain:
#   entrypoint-base.sh  →  entrypoint.sh  →  node starts
exec /entrypoint-base.sh "$@"
```

The chain becomes:

```
tini → /my-entrypoint.sh  (your custom logic)
                         → /entrypoint-base.sh  (UID/GID remapping, privilege drop)
                                               → /entrypoint.sh  (node init + start)
```

Replace `<version-tag>` with the desired image tag (e.g. `latest`) and `<your-package>` with the package(s) you need.

## Torubleshooting

**Fed-BioMed Docker images automatically generate configuration files and continue using them across container restarts.** These configuration files are stored in the mounted host directory corresponding to the container's `/fbm-node` path.

If the node component fails to connect to the researcher, it may be due to issues such as incorrect network configuration. In such cases, please ensure that any changes you make are applied directly to the configuration files located in the mounted directory on the host. Otherwise, the changes will not persist or be reflected inside the container.



