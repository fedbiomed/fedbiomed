ARG CONTAINER_GID=1122
ARG CONTAINER_UID=101
ARG CONTAINER_GROUP=fedbio
ARG CONTAINER_USER=fedbio-men


# Multi-stage nodejs
FROM node:20-bullseye AS node-base

# Final python installation stage
FROM python:3.10-slim-bullseye AS python-base

# Merge multi-state builds
FROM debian:bullseye
COPY --from=node-base /usr/local /usr/local
COPY --from=python-base /usr/local /usr/local



RUN apt-get update && apt-get install -y git build-essential curl

# Install entrypoint script
COPY ./envs/vpn/docker/base/build_files/entrypoint_functions.bash /


# Image build arguments for user name and group
ARG CONTAINER_GID
ARG CONTAINER_UID
ARG CONTAINER_GROUP
ARG CONTAINER_USER


# Create env variable for corresponding build arguments
ENV CONTAINER_UID=$CONTAINER_UID
ENV CONTAINER_GID=$CONTAINER_GID
# alpine does not accept users/groups with numerical names
ENV CONTAINER_USER=$CONTAINER_USER
ENV CONTAINER_GROUP=$CONTAINER_GROUP

# Verify installations
RUN node -v && npm -v  && python --version && pip --version
RUN npm install --force --global yarn@1.22

# Install necessary binaries
RUN apt-get update && apt-get install -y iptables iproute2 iputils-ping \
	bash vim net-tools procps build-essential kmod \
	python3-tk tk-dev libgmp3-dev libmpfr-dev libmpc-dev apt-utils wget git

# Setup subset of fedbiomed code tree
RUN mkdir -p /fedbiomed/envs /fbm-node /fbm-researcher

# Copy files from fedbiomed repo - in the context of fedbiomed top directory
COPY envs/vpn /fedbiomed/envs/vpn
COPY envs/common /fedbiomed/envs/common
COPY envs/common /fedbiomed/envs/common_reference
COPY fedbiomed /fedbiomed/fedbiomed
COPY notebooks /fedbiomed/notebooks
COPY docs/tutorials /fedbiomed/docs/tutorials
COPY scripts /fedbiomed/scripts
COPY fedbiomed_gui /fedbiomed/fedbiomed_gui
COPY pyproject.toml /fedbiomed/pyproject.toml
COPY README.md /fedbiomed/README.md
COPY hatch_build.py /fedbiomed/hatch_build.py

# Create and activate fedbiomed user
RUN groupadd --gid $CONTAINER_GID $CONTAINER_GROUP
RUN useradd -m --shell /bin/bash -d /home/$CONTAINER_USER \
	--uid $CONTAINER_UID --gid $CONTAINER_GID $CONTAINER_USER

# Prepare for work in context of alternate account
RUN chown -R $CONTAINER_USER:$CONTAINER_GROUP /fedbiomed
RUN chown -R $CONTAINER_USER:$CONTAINER_GROUP /fbm-node
RUN chown -R $CONTAINER_USER:$CONTAINER_GROUP /fbm-researcher

WORKDIR /fedbiomed

# Activate Fed-BioMed user
USER $CONTAINER_USER:$CONTAINER_GROUP

# Do full installation of Fed-BioMed
RUN pip install '.[node, researcher, gui]'

# Add local binary to PATH variable
ENV PATH=/home/${CONTAINER_USER}/.local/bin:$PATH

# transmit build-time values to running container
ENV CONTAINER_BUILD_USER=${CONTAINER_USER}
ENV CONTAINER_BUILD_GROUP=${CONTAINER_GROUP}
ENV CONTAINER_BUILD_UID=${CONTAINER_UID}
ENV CONTAINER_BUILD_GID=${CONTAINER_GID}

# get wireguard from builder image
# COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
# COPY --from=builder /usr/bin/wg* /usr/bin/

