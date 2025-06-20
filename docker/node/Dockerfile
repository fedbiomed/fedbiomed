# Version tag of Fed-BioMed image
ARG FBM_IMAGE_VERSION

FROM fedbiomed/base:${FBM_IMAGE_VERSION:-latest} AS fedbiomed-base

# Copy node entry point file
COPY docker/node/entrypoint.bash /entrypoint.bash
COPY docker/node/supervisord-node.conf.template /etc/supervisor/supervisord.conf.template

USER root:root

# Create node directory
RUN mkdir -p /fbm-node /data
RUN chown -R $FEDBIOMED_USER:$FEDBIOMED_GROUP /fbm-node /data

# Change working directory to fedbiomed
WORKDIR /fedbiomed

# Globally install fedbiomed
RUN pip install '.[node, gui]'

WORKDIR /
# Run node
ENTRYPOINT ["/entrypoint.bash"]
