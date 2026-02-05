# Multi-stage nodejs
FROM node:20-bullseye AS node-build

# Final python installation stage
FROM python:3.10-slim-bullseye AS python-build

# Merge multi-state builds
FROM debian:bullseye

ARG FEDBIOMED_GID=1001
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
	  tini \
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

# Create fedbiomed user first
RUN groupadd --gid $FEDBIOMED_GID $FEDBIOMED_GROUP && \
    useradd -m --shell /bin/bash -d /home/$FEDBIOMED_USER \
	--uid $FEDBIOMED_UID --gid $FEDBIOMED_GID $FEDBIOMED_USER

# Create log directory with correct ownership
RUN mkdir -p /home/$FEDBIOMED_USER/log && \
    chown $FEDBIOMED_USER:$FEDBIOMED_GROUP /home/$FEDBIOMED_USER/log && \
    chmod -R g+rwX /home/$FEDBIOMED_USER

# Copy entrypoint bash as root (needs to be executable by root)
COPY docker/base/entrypoint-base.sh /entrypoint-base.sh
RUN chmod +x /entrypoint-base.sh

# Copy files with correct ownership from the start
# COPY automatically creates parent directories
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP envs/vpn /home/${FEDBIOMED_USER}/fedbiomed/envs/vpn
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP envs/common /home/${FEDBIOMED_USER}/fedbiomed/envs/common
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP envs/common /home/${FEDBIOMED_USER}/fedbiomed/envs/common_reference
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP fedbiomed /home/${FEDBIOMED_USER}/fedbiomed/fedbiomed
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP notebooks /home/${FEDBIOMED_USER}/fedbiomed/notebooks
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP docs/tutorials /home/${FEDBIOMED_USER}/fedbiomed/docs/tutorials
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP scripts /home/${FEDBIOMED_USER}/fedbiomed/scripts
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP fedbiomed_gui /home/${FEDBIOMED_USER}/fedbiomed/fedbiomed_gui
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP pyproject.toml /home/${FEDBIOMED_USER}/fedbiomed/pyproject.toml
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP README.md /home/${FEDBIOMED_USER}/fedbiomed/README.md
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP hatch_build.py /home/${FEDBIOMED_USER}/fedbiomed/hatch_build.py

WORKDIR /home/${FEDBIOMED_USER}/fedbiomed

# Switch to non-root user before pip install
USER $FEDBIOMED_USER

# Add .local/bin to PATH for pip-installed commands
ENV PATH="/home/${FEDBIOMED_USER}/.local/bin:${PATH}"

# Install fedbiomed as the user
RUN pip install '.[ ]'

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["/entrypoint-base.sh"]
