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
docker exec -it -u fedbiomed my-node bash
```

!!! warning "Test" 
    Make sure to replace `fedbiomed` with the appropriate container user if it's different.


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

If SSL is activated, the Node GUI will use port `8443`; otherwise, it will use port `8484`.

You can choose the host port for this application. The following command will make the Fed-BioMed Node GUI accessible on port `8812`:

```bash
docker run -it \
    -v <path-to-host-fbm-node>:/fbm-node \
    -v <path-to-host-data>:/data \
    -e SSL_ON=True \
    -p 8812:8483
    fedbiomed/node-gui:latest
```

!!! note "Node GUI Ports"
    The Node GUI image exposes ports `8484` (non-SSL) and `8443` (SSL). If you launch multiple Node GUI instances, please use `-p <port>:8484` or `-p <port-ssl>:8443` to forward the container port to an available port on the host machine.

---

## Researcher

As it is in Node image researcher node work in a same way to lancuhs and configure. Environment variables can be used to manupulate researcher configuration or researcher configuration file located in the mounted directory can changed. 

The following commad will launch a researcher component with a basic configuration. There it is recommanded to mount the volume for `/fbm-researcher` directory in host that generated and keeps researcher component configurations, folders and files. Researcher component exposes a port that allows other component. Researcher component also exposes port tensorbard application to display feedback scalar values collected from particpating node during the training.  

```bash
docker run -it -d \ 
    -v <path-fedbiomed-researcher>:/fbm-researcher \
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


One of the advantages of Docker is that it allows you to define and manage multiple containers using a single **Docker Compose** file. This makes it easy to orchestrate several services at once, rather than launching each container manually.

This is particularly useful for **Fed-BioMed**, where you may want to test or run multiple nodes, Node GUIs, and a Researcher container simultaneously. We highly recommend using Docker Compose to simplify the setup and management of your Fed-BioMed containers.

Below is an example of a Docker Compose file that launches **two nodes**, **two Node GUI instances**, and **one Researcher**, ready to run a federated learning experiment.


```yaml
version: "3.9"

services:
  researcher:
    image: fedbiomed/researcher:latest
    container_name: researcher
    environment:
      - FBM_SERVER_HOST=0.0.0.0
      - FBM_SERVER_PORT=50051
    volumes:
      - ./my-researcher:/fbm-researcher
    ports:
      - "50051:50051"
    networks:
      - fedbiomed-net

  node1:
    image: fedbiomed/node:latest
    container_name: node1
    environment:
      - RESEARCHER_HOST=researcher
      - NODE_ID=node1
    volumes:
      - ./node1:/fbm-node
    networks:
      - fedbiomed-net
    depends_on:
      - researcher

  node1-gui:
    image: fedbiomed/node-gui:latest
    container_name: node1-gui
    volumes:
      - 
    environment:
      - SSL_ON=True
    ports:
      - "8441:8443"    # SSL GUI
    depends_on:
      - node1

  node2:
    image: fedbiomed/node:latest
    container_name: node2
    environment:
      - RESEARCHER_HOST=researcher
    volumes:
      - ./node2:/fbm-node
    networks:
      - fedbiomed-net
    depends_on:
      - researcher

  node2-gui:
    image: fedbiomed/node-gui:latest
    container_name: node2-gui
    volumes:
        - ./node2:/fbm-node
    environment:
      - SSL_ON=True 
    ports:
      - "8482:8484"
      - "8442:8443"
    depends_on:
      - node2

networks:
  fedbiomed-net:
    driver: bridge

```


After creating the `docker-compose.yml` file, navigate to the directory where the file is located and run:

```bash
docker compose up -d
```

This command will start all defined services in detached mode and mount the directories relative to the current working directory (where the command is executed).

You can check status of all lancuehd services using the following command. It will list all active containers, including their names, ports, and uptime.

```
docker ps
```

## Building Docker Images from Source

If desired, it is possible to build Fed-BioMed Docker images from source with custom modifications, such as changing the default user. This process is described in the [building Fed-BioMed Docker image documentation](../../developer/docker-images.md) for developers.

## Extending Docker Images 

As is the nature of Docker, you can always use the **Fed-BioMed Docker images as a base** to extend their capabilities or customize them for specific use cases or deployment scenarios. You can explore the image definitions located in the `docker/` directory of the Fed-BioMed source repository to see what has been installed and configured in the base images.

This makes it easy to tailor the Fed-BioMed setup — for example, by adding additional packages, integrating custom extensions, or embedding Fed-BioMed within a larger framework.

Here is an example of a custom `Dockerfile` that extends the Node image:

```Dockerfile
FROM fedbiomed/node:<version-tag>

# Install additional Python packages
RUN pip install <a-package-required-for-your-setup>

# Add your custom Docker instructions here...

ENTRYPOINT ["/entrypoint.bash"] # Optional
```

Replace `<version-tag>` with the desired image tag (e.g., `latest`) and `<a-package-required-for-your-setup>` with the package(s) you need.

## Torubleshooting

**Fed-BioMed Docker images automatically generate configuration files and continue using them across container restarts.** These configuration files are stored in the mounted host directory corresponding to the container's `/fbm-node` path.

If the node component fails to connect to the researcher, it may be due to issues such as incorrect network configuration. In such cases, please ensure that any changes you make are applied directly to the configuration files located in the mounted directory on the host. Otherwise, the changes will not persist or be reflected inside the container.




