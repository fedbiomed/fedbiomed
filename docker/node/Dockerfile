# Version tag of Fed-BioMed image
ARG FBM_IMAGE_VERSION

FROM fedbiomed/base:${FBM_IMAGE_VERSION:-latest} AS fedbiomed-base

# Copy node entry point file
COPY docker/node/entrypoint.bash /entrypoint.bash
COPY docker/node/supervisord-node.conf.template /etc/supervisor/supervisord.conf.template

USER root:root

# Create node directory - ownership will be set at runtime to support dynamic user mapping
# Note: /fbm-node ownership will be set in entrypoint.bash to support runtime user mapping
RUN mkdir -p /fbm-node

# Change working directory to fedbiomed
WORKDIR /fedbiomed

# Globally install fedbiomed
RUN pip install '.[node]'

# Restore ownership of /fedbiomed after pip install (which runs as root)
RUN chown -R $FEDBIOMED_USER:$FEDBIOMED_GROUP /fedbiomed

WORKDIR /
# Run node
ENTRYPOINT ["/entrypoint.bash"]
