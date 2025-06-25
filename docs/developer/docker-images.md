# Building Docker Images from Source

This documentation explains how to build the Fed-BioMed Docker images from scratch. It will guide you through the necessary steps to create each image, customize build parameters such as user settings and version tags, and ensure proper version compatibility across the components. Whether you are building images for development, customization, or deployment, this guide will help you get started.

## Building Images

Docker images are located in the `docker` directory at the root of the Fed-BioMed repository. This directory contains subdirectories, each corresponding to a different Fed-BioMed component:

* **Base:** A minimal Docker image with the core Fed-BioMed installation and essential binaries.
* **Node:** The Docker image for the Fed-BioMed Node component.
* **Node GUI:** The image for the Fed-BioMed Node web application, including all necessary modules for the web server.
* **Node GPU:** A Node image preconfigured with the required modules to enable GPU-based training.
* **Researcher:** The Docker image for the Fed-BioMed Researcher component.

!!! warning "Build order" 
    Docker images depend on each other, so it is important to respect the correct build order. The Base Docker image must be built first. After that, you can build the Node, Researcher, and Node GUI images. Once the Node image is built, the Node GPU image can be built on top of it. 

### Fed-BioMed Base Image

First, please clone the Fed-BioMed repository and navigate to its root directory. All Docker build commands should be executed from this directory. The example below builds the base Docker image with the default user and settings:

```bash
git clone https://github.com/fedbiomed/fedbiomed.git
cd fedbiomed
docker build -t fedbiomed/base:<tag> . -f docker/base/Dockerfile 
```

In some cases, you may want to customize the default user and group in the image. You can do this using build arguments as shown below:

```bash
docker build \
    --build-arg FEDBIOMED_USER=<user-name> \
    --build-arg FEDBIOMED_UID=<user-id> \
    --build-arg FEDBIOMED_GROUP=<group-name> \
    --build-arg FEDBIOMED_GID=<group-id> \
    -t fedbiomed/base:<tag> ./ -f docker/base/Dockerfile
```

Once the base image is built with a custom user, all other Fed-BioMed Docker images that use this base image will inherit that user configuration (except Node GPU). Therefore, if you change the user in the base image, make sure to rebuild the other images accordingly to ensure consistency across the stack. However, this is not the case for the Node GPU image, because it uses a multi-stage build where the base image is an NVIDIA-provided image. To rebuild a Node GPU image with a different user, you must re-declare the build arguments in the build command.

!!! info "Tags"
    Please note that image tags are important to ensure that other images using the base image reference the correct version. By default, in the docker images the tag `latest` is used. Therefore, please use `latest` tag for `<tag>` if you want to follow default configuration.However, we recommend assigning appropriate and consistent tags when building images.


### Building Node Images

As it is mentioned above, when building Docker images for Fed-BioMed components, it is essential to use consistent and appropriate tags across all images. The `FBM_IMAGE_VERSION` build argument allows you to specify which version (i.e., tag) of the base image should be used during the build process.

By default, if `FBM_IMAGE_VERSION` is not provided, the tag `latest` will be used:

```Dockerfile
FROM fedbiomed/base:${FBM_IMAGE_VERSION:-latest}
```

This means that if you have built the base image with the tag `latest`, you don't need to explicitly set `--build-arg FBM_IMAGE_VERSION=latest`. However, if you used a custom tag (e.g., `my-tag-or-version`), you **must** pass the same tag to the dependent image builds to ensure compatibility.

#### Building the Node Image

```bash
docker build \
    --build-arg FBM_IMAGE_VERSION=<tag> \
    -t fedbiomed/node:<tag> ./ -f docker/node/Dockerfile
```

#### Building the Node GUI Image

```bash
docker build \
    --build-arg FBM_IMAGE_VERSION=<tag> \
    -t fedbiomed/node-gui:<tag> ./ -f docker/node-gui/Dockerfile
```

#### Building the Node GPU Image

To build the `fedbiomed/node-gpu` image, you can use the following command:

```bash
docker build \
    --build-arg FBM_IMAGE_VERSION=<tag> \
    -t fedbiomed/node-gpu:<tag> ./ -f docker/node-gpu/Dockerfile
```

However, this is **not sufficient** when you want to customize the image for a different user. This is because the Node GPU image uses a multi-stage build where the base image is provided by NVIDIA, and user-related configurations are not automatically inherited.

To rebuild the Node GPU image with a different user, you **must explicitly re-declare** the following build arguments:

```bash
docker build \
    --build-arg FBM_IMAGE_VERSION=<tag> \
    --build-arg FEDBIOMED_USER=<user-name> \
    --build-arg FEDBIOMED_UID=<user-id> \
    --build-arg FEDBIOMED_GROUP=<group-name> \
    --build-arg FEDBIOMED_GID=<group-id> \
    -t fedbiomed/node-gpu:<tag> ./ -f docker/node-gpu/Dockerfile
```

These arguments ensure that the user and group setup from the base image are correctly replicated in the final stage of the build.

### Building the Researcher Image

Just like the node images, the researcher image depends on the base image tag. Ensure you pass the correct tag using the `FBM_IMAGE_VERSION` build argument:

```bash
docker build \
    --build-arg FBM_IMAGE_VERSION=<tag> \
    -t fedbiomed/researcher:<tag> ./ -f docker/researcher/Dockerfile
```


After all images are built, please refer to the [Fed-BioMed Docker deployment documentation](../user-guide/deployment/docker.md) for guidance on how to run and configure these containers.

