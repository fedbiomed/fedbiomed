FROM fedbiomed/vpn-base

# jupyter notebook
EXPOSE 8888
# tensorboard via socat proxy, should be exposed as 6006
EXPOSE 6007


ARG CONTAINER_GID
ARG CONTAINER_UID
ARG CONTAINER_GROUP
ARG CONTAINER_USER
ARG PYTHON_VERSION="3.10"

# assign default values even when variable exists and is empty
# (ARG default value only applies when variable is non existent)
# + make it persistent at runtime (pass this variable to container)
ENV CONTAINER_UID=${CONTAINER_UID:-0}
ENV CONTAINER_GID=${CONTAINER_GID:-0}
# alpine does not accept users/groups with numerical names
ENV CONTAINER_USER=${CONTAINER_USER:-root}
ENV CONTAINER_GROUP=${CONTAINER_GROUP:-root}

RUN apt-get update && apt-get install -y libgmp3-dev openssl \
	libreadline-dev xz-utils libncurses-dev libffi-dev \
	libmpfr-dev libmpc-dev apt-utils wget git \
	python-tk python${PYTHON_VERSION} python3-pip

# Create alias
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip


# read config.env in interactive shells & setup for reading it in entrypoint
COPY ./envs/vpn/docker/researcher/build_files/bashrc_append /tmp
RUN cat /tmp/bashrc_append | tee -a /root/.bashrc /root/bashrc_entrypoint \
        /etc/skel/.bashrc /etc/skel/bashrc_entrypoint >/dev/null

RUN [ "$CONTAINER_GROUP" != 'root' ] && groupadd -g $CONTAINER_GID $CONTAINER_GROUP || :
RUN [ "$CONTAINER_USER" != 'root' ] && useradd -m -d /home/$CONTAINER_USER \
        -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_USER || :

COPY ./envs/vpn/docker/researcher/build_files/entrypoint*.bash /

# Setup subset of fedbiomed code tree
RUN mkdir -p /fedbiomed/envs /data
RUN mkdir -p /fbm-researcher

# Copy files from fedbiomed repo - in the context of fedbiomed top directory
COPY envs/vpn /fedbiomed/envs/vpn
COPY envs/common/default_biprimes /fedbiomed/envs/common/default_biprimes
COPY fedbiomed /fedbiomed/fedbiomed
COPY scripts /fedbiomed/scripts
COPY notebooks /fedbiomed/notebooks
COPY docs/tutorials /fedbiomed/docs/tutorials
COPY gui /fedbiomed/gui
COPY pyproject.toml /fedbiomed/pyproject.toml
COPY README.md /fedbiomed/README.md

# Alternative option : git clone & strip unused dirs
#  - ensures working with clean repo (docker build begins with a stat(), hangs for
#    a long time if you have zillion files in notebooks/data)
#  - ensures using develop branch (good or bad)

# Prepare for work in context of alternate account
RUN chown -R $CONTAINER_USER:$CONTAINER_GROUP /fedbiomed
RUN chown -R $CONTAINER_USER:$CONTAINER_GROUP /fbm-researcher
WORKDIR /fedbiomed

# Install fedbiomed and its requirements
RUN pip install '.[researcher]'
RUN pip install jupyter

# transmit build-time values to running container
ENV CONTAINER_BUILD_USER=${CONTAINER_USER}
ENV CONTAINER_BUILD_GROUP=${CONTAINER_GROUP}
ENV CONTAINER_BUILD_UID=${CONTAINER_UID}
ENV CONTAINER_BUILD_GID=${CONTAINER_GID}
ENV FBM_RESEARCHER_COMPONENT_ROOT=/fbm-researcher

# Activates fedbiomed command for root too.
ENV PATH="$HOME/.local/bin:$PATH"


WORKDIR /

# Launch as root to manage VPN
USER root:root
ENTRYPOINT ["/entrypoint.bash"]

