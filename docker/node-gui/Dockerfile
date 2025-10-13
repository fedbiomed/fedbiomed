ARG FBM_IMAGE_VERSION

# Start from Fed-BioMed base image
FROM fedbiomed/node:${FBM_IMAGE_VERSION:-latest} AS fedbiomed-node

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



# Copy node-gui entry point file
COPY docker/node-gui/entrypoint.bash /entrypoint-gui.bash
COPY docker/node-gui/nginx /fedbiomed/nginx

USER root:root
# Install socat
RUN apt-get update && apt-get install -y socat

# Install Nginx along with the missing dependencies
RUN apt-get install nginx -y --fix-missing


# Create node directory
RUN mkdir /certs
RUN chown -R $FEDBIOMED_USER:$FEDBIOMED_GROUP /certs 

# Change working directory to fedbiomed
WORKDIR /fedbiomed

# Install only GUI dependencies
RUN pip install '.[gui]'

# Restore ownership of /fedbiomed after pip install (which runs as root)
RUN chown -R ${FEDBIOMED_USER}:${FEDBIOMED_GROUP} /fedbiomed

WORKDIR /

# Run researcher
ENTRYPOINT ["/entrypoint-gui.bash"]
