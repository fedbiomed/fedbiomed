# Start from Fed-BioMed base image
ARG FBM_IMAGE_VERSION

FROM fedbiomed/base:${FBM_IMAGE_VERSION:-latest} AS fedbiomed-base

# jupyter notebook
EXPOSE 8888
# tensorboard via socat proxy, should be exposed as 6006
EXPOSE 6007

# Copy node entry point file
COPY docker/researcher/entrypoint.bash /entrypoint.bash

USER root:root
# Install socat
RUN apt-get update && apt-get install -y socat

# Create node directory
RUN mkdir -p /fbm-researcher
RUN chown -R $FEDBIOMED_USER:$FEDBIOMED_GROUP /fedbiomed

# Change working directory to fedbiomed
WORKDIR /fedbiomed

# Globally install fedbiomed
RUN pip install '.[researcher]'

WORKDIR /

# Run researcher
ENTRYPOINT ["/entrypoint.bash"]
