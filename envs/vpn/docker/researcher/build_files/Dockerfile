ARG FBM_CONTAINER_VERSION_TAG

FROM fedbiomed/base:${FBM_CONTAINER_VERSION_TAG:-unknown}

# jupyter notebook
EXPOSE 8888
# tensorboard via socat proxy, should be exposed as 6006
EXPOSE 6007

# Root actions
USER root:root

COPY ./envs/vpn/docker/researcher/build_files/entrypoint.bash /

# Remove instantiated fbm-node directory
RUN rm -r /fbm-node

# Create a directory for researcher
RUN mkdir -p /fbm-researcher

RUN ln -s /fedbiomed/docs/tutorials /fedbiomed/notebooks/tutorials

RUN chown -R $CONTAINER_USER:$CONTAINER_GROUP /fbm-researcher

# Set necessary env variables
ENV FBM_RESEARCHER_COMPONENT_ROOT=/fbm-researcher

# Launch as root to manage VPN
WORKDIR /
ENTRYPOINT ["/entrypoint.bash"]

