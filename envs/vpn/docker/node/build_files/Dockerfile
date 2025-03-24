# Use base fedbiomed image
FROM fedbiomed/base::${FBM_CONTAINER_VERSION_TAG:-latest}
USER root:root

# read config.env in interactive shells & setup for reading it in entrypoint
COPY ./envs/vpn/docker/node/build_files/bashrc_append /tmp
RUN cat /tmp/bashrc_append | tee -a /root/.bashrc /root/bashrc_entrypoint \
        /etc/skel/.bashrc /etc/skel/bashrc_entrypoint >/dev/null

# Copy entrypoint scripts
COPY ./envs/vpn/docker/node/build_files/entrypoint*.bash /

USER $CONTAINER_USER:$CONTAINER_GROUP
WORKDIR /fedbiomed

# Set container user
USER $CONTAINER_USER:$CONTAINER_GROUP

# Installs fedbiomed and necessary modules
RUN /bin/pip install --no-warn-script-location -I '.[node]'

# transmit build-time values to running container
ENV CONTAINER_BUILD_USER=${CONTAINER_USER}
ENV CONTAINER_BUILD_GROUP=${CONTAINER_GROUP}
ENV CONTAINER_BUILD_UID=${CONTAINER_UID}
ENV CONTAINER_BUILD_GID=${CONTAINER_GID}

# Changedir to main directory
WORKDIR /

# Launch as root to manage VPN
USER root:root
ENTRYPOINT ["/entrypoint.bash"]

