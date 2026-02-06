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
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP docker/researcher/entrypoint.sh /entrypoint.sh

USER root:root
# Install socat
RUN apt-get update && apt-get install -y socat

# Create /fbm-researcher directory with proper ownership
RUN mkdir -p /fbm-researcher && \
    chown -R ${FEDBIOMED_USER}:${FEDBIOMED_GROUP} /fbm-researcher

USER ${FEDBIOMED_USER}:${FEDBIOMED_GROUP}

# Change working directory to fedbiomed
WORKDIR /home/${FEDBIOMED_USER}/fedbiomed

# Install only researcher dependencies 
RUN pip install --user --no-cache-dir --no-index \
    --find-links=/tmp/wheels \
    'fedbiomed[researcher]'

WORKDIR /home/${FEDBIOMED_USER}/

# Note: Uses CMD from base image which runs /entrypoint-base.sh
# /entrypoint-base.sh will execute /entrypoint.sh (provided by this image)
