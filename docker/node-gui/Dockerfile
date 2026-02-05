ARG FBM_IMAGE_VERSION

# Start from Fed-BioMed base image
FROM fedbiomed/node:${FBM_IMAGE_VERSION:-latest} AS fedbiomed-node

# Redeclare ARGs from base image to make them available at build time
ARG FEDBIOMED_USER=fedbiomed
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_UID=1001
ARG FEDBIOMED_GID=1001

# Set environment variables to make them available at runtime
ENV FEDBIOMED_USER=${FEDBIOMED_USER}
ENV FEDBIOMED_GROUP=${FEDBIOMED_GROUP}
ENV FEDBIOMED_UID=${FEDBIOMED_UID}
ENV FEDBIOMED_GID=${FEDBIOMED_GID}

ARG NGINX_PORT_SSL=8483
ARG NGINX_PORT_NOSSL=8484

ENV NGINX_PORT_SSL=${NGINX_PORT_SSL}
ENV NGINX_PORT_NOSSL=${NGINX_PORT_NOSSL}

# Gui will use mounted fbm-node directory to get node configuration
VOLUME /fbm-node
# Data directory where data files are kept
VOLUME /data
# GUI SSL certificates
VOLUME /certs 

# nginx + flask HTTP server for node1
EXPOSE 8484
# nginx + gunicorn HTTPS server for node1
EXPOSE 8443


USER root:root
# Install socat
RUN apt-get update && apt-get install -y socat

# Install Nginx along with the missing dependencies
RUN apt-get install nginx -y --fix-missing

# Copy node-gui entry point files
# entrypoint-root.sh runs as root before privilege drop (nginx config)
COPY docker/node-gui/entrypoint-root.sh /entrypoint-root.sh
RUN chmod +x /entrypoint-root.sh

# entrypoint.sh runs as user after privilege drop (GUI app)
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} docker/node-gui/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} docker/node-gui/nginx /home/${FEDBIOMED_USER}/nginx

USER ${FEDBIOMED_USER}:${FEDBIOMED_GROUP}

RUN mkdir -p /home/${FEDBIOMED_USER}/nginx/logs && \
    chmod -R 700 /home/${FEDBIOMED_USER}/nginx

# Create node directory
RUN mkdir /home/${FEDBIOMED_USER}/certs

# Change working directory to fedbiomed
WORKDIR /home/${FEDBIOMED_USER}/fedbiomed

# Install only GUI dependencies
RUN pip install '.[gui]'

# Nginx requires write access to its log and run 
# directories, so  we set those up with the correct 
# ownership and permissions
USER root:root


WORKDIR /home/${FEDBIOMED_USER}/

