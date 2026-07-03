FROM ghcr.io/wg-easy/wg-easy:latest

# Fedbiomed default users
ARG FEDBIOMED_USER=fedbiomed
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_UID=2101
ARG FEDBIOMED_GID=2101

# Set environment variables to make them available at runtime
ENV FEDBIOMED_USER=${FEDBIOMED_USER}
ENV FEDBIOMED_GROUP=${FEDBIOMED_GROUP}
ENV FEDBIOMED_UID=${FEDBIOMED_UID}
ENV FEDBIOMED_GID=${FEDBIOMED_GID}

# Default folders and users in every Fed-BioMed container
# Important: uses Alpine user/group tools (addgroup/adduser, not groupadd/useradd)
RUN addgroup -g $FEDBIOMED_GID $FEDBIOMED_GROUP && \
    adduser -D -h /home/$FEDBIOMED_USER -s /bin/bash -u $FEDBIOMED_UID -G $FEDBIOMED_GROUP $FEDBIOMED_USER && \
    mkdir -p /home/$FEDBIOMED_USER/fedbiomed && \
    mkdir -p /home/$FEDBIOMED_USER/log && \
    chown -R $FEDBIOMED_USER:$FEDBIOMED_GROUP /home/$FEDBIOMED_USER

# Change user to root for package installation
USER root:root

RUN apk add --no-cache curl jq

# Install Python 3 + pip on Alpine
RUN apk add --no-cache python3 py3-pip

# If you need to pip install packages, Alpine's system Python is externally-managed,
# so use a venv or --break-system-packages
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN /opt/venv/bin/pip install --no-cache-dir -qq requests tabulate

# Copy in custom scripts, configs, etc.
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP docker/vpn/vpn-server/custom-scripts/ /custom-scripts/
RUN chmod +x /custom-scripts/*.sh


COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP docker/vpn/vpn-server/vpn /vpn
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP docker/base/entrypoint-base.sh /fedbiomed-entrypoint-base.sh
