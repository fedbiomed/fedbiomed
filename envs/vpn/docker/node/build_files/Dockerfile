# Use base fedbiomed image
ARG version

FROM fedbiomed/base:${version:-latest}
USER root:root

# read config.env in interactive shells & setup for reading it in entrypoint
COPY ./envs/vpn/docker/node/build_files/bashrc_append /tmp
RUN cat /tmp/bashrc_append | tee -a /root/.bashrc /root/bashrc_entrypoint \
        /etc/skel/.bashrc /etc/skel/bashrc_entrypoint >/dev/null

# Copy entrypoint scripts
COPY ./envs/vpn/docker/node/build_files/entrypoint.bash /

# Changedir to main directory
WORKDIR /

# Launch as root to manage VPN
USER $CONTAINER_USER:$CONTAINER_GROUP
ENTRYPOINT ["/entrypoint.bash"]

