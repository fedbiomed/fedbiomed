# Deploying Fed-BioMed Using Docker Image

## Introduction

Fed-BioMed provides a set of Docker images designed to simplify the deployment, configuration, and testing of its core components. These images are intended to help users get started quickly, without the need to manually install dependencies or configure environments for each component. Whether you're a researcher experimenting with federated learning workflows or a developer integrating Fed-BioMed into a larger system, these Docker images serve as a ready-to-use components of Fed-BioMed.

The Docker images not only support the standard use cases but are also designed to be easily extendable. You can customize or build on top of the base images to suit their specific requirements, such as adding wrapping Fed-BioMed within your application, integrating new data sources, or generally adapting the behavior of components like the node or researcher to align with your infrastructure.

This documentation will guide you through the available Docker images for each Fed-BioMed component, explain how to launch and manage them, and provide best practices for extending and customizing the containers to fit your needs.
 
## Pulling Docker Images 

Fed-BioMed Docker images are published for each released version of Fed-BioMed.
You can visit [Docker Hub](https://hub.docker.com/u/fedbiomed) to see the available images published by the Fed-BioMed team.

!!! warning "Component Versions"
    Although Fed-BioMed components are generally backward compatible, it is recommended to use the same version across all component images to ensure full compatibility and stability.

---

### Node 

The following command pulls the Fed-BioMed Node image and runs it. Running this image will automatically start a Fed-BioMed Node component.

!!! note "Running in background"
    Please add the `-d` option to run the Docker container in the background.

```bash
docker run -it  \
    --name my-node \
    --network host \
    -v <absolute-path-to-host-fbm-node>:/fbm-node \
    -e FBM_SECURITY_SECURE_AGGREGATION=True \
    -e FBM_RESEARCHER_PORT=50051 \
    -e FBM_RESEARCHER_IP=localhost \
    fedbiomed/node:latest
```

For testing purposes, it's fine to use the default configuration. However, for deployment, you may want to review the available configuration options to ensure the Node is properly set up. Component configuration can be managed using environment variables. All environment variables listed in the [Node configuration guide](../nodes/configuring-nodes.md) can be passed to the container using Docker’s `--env` option. 

It is recommended to update the configuration directly in the file located at `<absolute-path-to-host-fbm-node>/etc/config.ini`. This approach provides a more stable and persistent setup. Please see the section [Configuration of Node Container](#configuration-of-node-container) for more details.


Keep in mind that any environment variables set at runtime will always override the values defined in config.ini.


#### Launching docker container with a diffrent user

By default, the Fed-BioMed Node component is launched using a predefined user inside the Docker container. However, you can specify a different user at runtime to avoid permission issues when working with files on your local machine.

Here's how to run the container with custom user settings:

```bash
docker run -it \
    --name my-node \
    -v <path-to-local-fbm-node>:/fbm-node \
    -e CONTAINER_USER=<user-name> \
    -e CONTAINER_UID=<user-id> \
    -e CONTAINER_GROUP=<group-name> \
    -e CONTAINER_GID=<group-id> \
    fedbiomed/node:latest
```

Please, replace `<user-name>`, `<user-id>`, `<group-name>`, and `<group-id>` with the corresponding values from your host system.

#### Configuration of Node Container

Node configuration can be manipulated at runtime by assigning environment variables using the `--env` or `-e` flag. These variables will also define the initial configuration values during the first run of the container.

After the Docker container is started for the first time, the node configuration is created inside the container under `/fbm-node/data`. To persist this configuration across runs or to edit it manually, mount a local directory using the `-v` option:

```bash
-v <path-to-host-fbm-node>:/fbm-node
```

To modify the configuration, edit the file located at:

```
<absolute-path-to-host-fbm-node>/etc/config.ini
```

Then restart the Fed-BioMed node container:

```bash
docker restart my-node
```

---

##### Configuration Behavior Overview:

- **First run with environment variables**:
    If the container is run for the first time with environment variables, a new node configuration will be created using those values.

- **Using an existing configuration directory**:
    If a previously created node directory is mounted, the container will use the configuration found there.

- **Overriding existing configuration at runtime**:
    If a static configuration already exists but environment variables are still provided, those variables will override the config **only at runtime**, without modifying the actual `config.ini` file.




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
docker exec -it -u fedman my-node bash
```

!!! warning "Test" 
    Make sure to replace `fedman` with the appropriate container user if it's different.


##### 3. Registering the Dataset via CLI

Once inside the container, use the Fed-BioMed CLI to add the dataset:

```bash
$ fedbiomed node dataset add
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

### Node GUI

Node GUI can be launched seperated from node as long as the mounted Fed-BioMed Node directory are the same. Node GUI assumes that the Node container has aldeay lancuhed and component is initialized in the directory that is going to be mounted. Therefore, please make sure that the first launch Fed-BioMed docker conitaner in order to initialize Node component with desired configuration. 

```bash
docker run -it --name my-node-gui \
    -v <path-to-host-fbm-node>:/fbm-node \
    -p 8484:8484
    fedbiomed/node-gui:latest
```

!!! important "Volumes are mandatory"
    You must provide the `fbm-node` volume from the host where the Fed-BioMed node is instantiated.

Node GUI allows to activate SSL. If you want to activate SSL, please pass the following environment variable. You can setup a custom SSL certificate fby copying your certificates into `<path-to-host-fbm-node>/gui/certs` 

```bash
docker run -it \
    -v <path-to-local-fbm-node>:/fbm-node \
    -v <path-to-host-data>:/data \
    -e SSL_ON=True \
    fedbiomed/node-gui:latest
```

If SSL is activated, the Node GUI will use port `8483`; otherwise, it will use port `8484`.

You can choose the host port for this application. The following command will make the Fed-BioMed Node GUI accessible on port `8812`:

```bash
docker run -it \
    -v <path-to-host-fbm-node>:/fbm-node \
    -v <path-to-host-data>:/data \
    -e SSL_ON=True \
    -p 8812:8483
    fedbiomed/node-gui:latest
```

---

## Running Researcher container

The follwing commad will launch a researcher component with a basic configuration. There it is recommanded to mount the volume for `/fbm-researcher` directory in host that generated and keeps researcher component configurations, folders and files. Researcher component exposes a port that allows other component. Researcher component also exposes port tensorbard application to display feedback scalar values collected from particpating node during the training.  

```bash
docker run -it -d \ 
    -v <path-fedbiomed-researcher>:/fbm-researcher \
    -p 50051:50051 \
    fedbiomed/researcher:latest
```

### IP/Hostname of the Researcher component

```
docker run -it -d \
    -e FBM_SERVER_HOST=127.0.0.1 \
    -v <path-fedbiomed-researcher>:/fbm-researcher \
    -p 50051:50051 \
    fedbiomed/researcher:latest
```


## Use Docker Compose to Run and Manage Multiple Fed-BioMed Instances

Add docker compose example 


## Building Images

### Build base image

```bash
cd <fedbiomed-clone>
docker build -t fedbiomed/base:<tag> . -f docker/base/Dockerfile 
```

Define the Fed-BioMed user while building the image:

```bash
docker build \
    --build-arg FEDBIOMED_USER=<user-name> \
    --build-arg FEDBIOMED_UID=<user-id> \
    --build-arg FEDBIOMED_GROUP=<group-name> \
    --build-arg FEDBIOMED_GID=<group-id> \
    -t fedbiomed/base:<tag> . -f docker/base/Dockerfile
```

### Build node image

Once the user is created in the base image, there is no need to redefine it for subsequent images:

```bash
docker build -t fedbiomed/node:<tag> . -f docker/node/Dockerfile
```

