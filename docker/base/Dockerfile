# Multi-stage nodejs
FROM node:20-bullseye AS node-build

# Final python installation stage
FROM python:3.10-slim-bullseye AS python-build

# Merge multi-state builds
FROM debian:bullseye

ARG FEDBIOMED_GID=101
ARG FEDBIOMED_UID=1001
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_USER=fedbiomed

# Set environment variable to remember the user
ENV FEDBIOMED_GID=${FEDBIOMED_GID}
ENV FEDBIOMED_UID=${FEDBIOMED_UID}
ENV FEDBIOMED_GROUP=${FEDBIOMED_GROUP}
ENV FEDBIOMED_USER=${FEDBIOMED_USER}

# Install dependencies that will be needed for Node and Python
RUN apt-get update && apt-get install -y \
    ca-certificates libssl-dev \
    libffi-dev libbz2-dev \
    libjpeg-dev libpng-dev \
    liblzma-dev libreadline-dev libsqlite3-dev \
    gcc zlib1g python3-tk vim \
	  libgmp3-dev libmpfr-dev libmpc-dev \
	  gettext-base \
	  supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy Node
COPY --from=node-build /usr/local/ /usr/local/
COPY --from=node-build /opt/ opt/

# Copy Python
COPY --from=python-build /usr/local/ /usr/local/
COPY --from=python-build /usr/lib/ /usr/lib/

# Add symlinks for python and pip
#RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python && \
#    ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

# Setup subset of fedbiomed code tree
RUN mkdir -p /fedbiomed/envs

# Copy entrypoint bash
COPY docker/base/functions.bash /functions.bash

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
RUN groupadd --gid $FEDBIOMED_GID $FEDBIOMED_GROUP
RUN useradd -m --shell /bin/bash -d /home/$FEDBIOMED_USER \
	--uid $FEDBIOMED_UID --gid $FEDBIOMED_GID $FEDBIOMED_USER


# Create log directory
RUN mkdir -p /home/$FEDBIOMED_USER/log

# Prepare for work in context of alternate account
RUN chown -R $FEDBIOMED_USER:$FEDBIOMED_GROUP /fedbiomed /home/$FEDBIOMED_USER/log


# Give permission to Fed-BioMed group to allows user in the same group 
# can user fedbiomed home directory
RUN chgrp -R $FEDBIOMED_GROUP /home/$FEDBIOMED_USER
RUN chmod -R g+rwX /home/$FEDBIOMED_USER


WORKDIR /fedbiomed

# Install fedbiomed globally
RUN pip install '.[ ]'


# Switch to Fed-BioMed user
USER $FEDBIOMED_USER:$FEDBIOMED_GROUP

# Default command (optional)
CMD ["bash"]
