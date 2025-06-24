ARG FBM_IMAGE_VERSION

# Stage 1: base with image1 (e.g. fedbiomed-node)
FROM fedbiomed/node:${FBM_IMAGE_VERSION:-latest} AS fedbiomed-node

# Stage 2: base with image2 (e.g. nvidia/cuda)
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 AS cuda

ARG FEDBIOMED_USER=fedbiomed  
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_UID=1001
ARG FEDBIOMED_GID=101

ENV FEDBIOMED_USER=${FEDBIOMED_USER}
ENV FEDBIOMED_GROUP=${FEDBIOMED_GROUP}
ENV FEDBIOMED_UID=${FEDBIOMED_UID}
ENV FEDBIOMED_GID=${FEDBIOMED_GID}

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

COPY docker/node-gpu/before-entrypoint-hook.sh /before-entrypoint-hook.sh

# Copy installed Fed-BioMed Modules from fedbiomed image
COPY --from=fedbiomed-node /usr/local/ /usr/local/
COPY --from=fedbiomed-node /usr/lib/ /usr/lib/
COPY --from=fedbiomed-node /entrypoint.bash /entrypoint.bash
COPY --from=fedbiomed-node /functions.bash /functions.bash
COPY --from=fedbiomed-node /usr/lib/ /usr/lib/
COPY --from=fedbiomed-node /fbm-node /fbm-node
COPY --from=fedbiomed-node /etc/supervisor/supervisord.conf.template \
	/etc/supervisor/supervisord.conf.template

# Create and activate fedbiomed user
RUN groupadd --gid $FEDBIOMED_GID $FEDBIOMED_GROUP
RUN useradd -m --shell /bin/bash -d /home/$FEDBIOMED_USER \
	--uid $FEDBIOMED_UID --gid $FEDBIOMED_GID $FEDBIOMED_USER

# Create log directory
RUN mkdir -p /home/$FEDBIOMED_USER/log

# Give permission to Fed-BioMed group to allows user in the same group 
# can user fedbiomed home directory
RUN chgrp -R $FEDBIOMED_GROUP /home/$FEDBIOMED_USER
RUN chmod -R g+rwX /home/$FEDBIOMED_USER

# Restore entrypoint and working directory
WORKDIR /
ENTRYPOINT ["/entrypoint.bash"]










