# Start from Fed-BioMed base image
ARG FBM_IMAGE_VERSION

FROM fedbiomed/base:${FBM_IMAGE_VERSION:-latest} AS fedbiomed-base

# Redeclare ARGs from base image to make them available at build time
ARG FEDBIOMED_USER=fedbiomed
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_UID=1001
ARG FEDBIOMED_GID=101

# Set environment variables to make them available at runtime
ENV FEDBIOMED_USER=${FEDBIOMED_USER}
ENV FEDBIOMED_GROUP=${FEDBIOMED_GROUP}
ENV FEDBIOMED_UID=${FEDBIOMED_UID}
ENV FEDBIOMED_GID=${FEDBIOMED_GID}

# jupyter notebook
EXPOSE 8888
# tensorboard via socat proxy, should be exposed as 6006
EXPOSE 6007

# Copy node entry point file
COPY docker/researcher/entrypoint.bash /entrypoint.bash

USER root:root
# Install socat
RUN apt-get update && apt-get install -y socat

# Create researcher directory - ownership will be set at runtime to support dynamic user mapping
RUN mkdir -p /fbm-researcher
# Note: /fbm-researcher ownership will be set in entrypoint.bash to support runtime user mapping
RUN chown -R ${FEDBIOMED_USER}:${FEDBIOMED_GROUP} /fbm-researcher

# Change working directory to fedbiomed
WORKDIR /fedbiomed

# Globally install fedbiomed
RUN pip install '.[researcher]'

# Restore ownership of /fedbiomed after pip install (which runs as root)
RUN chown -R $FEDBIOMED_USER:$FEDBIOMED_GROUP /fedbiomed

WORKDIR /

# Run researcher
ENTRYPOINT ["/entrypoint.bash"]
