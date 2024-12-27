FROM python:3.10-slim-bullseye

# nginx + flask HTTP server for node1
EXPOSE 8484
# nginx + flask HTTP server for node2
EXPOSE 8485
# nginx + gunicorn HTTPS server for node1
EXPOSE 8443
# nginx + gunicorn HTTPS server for node2
EXPOSE 8444

ARG CONTAINER_GID
ARG CONTAINER_UID
ARG CONTAINER_GROUP
ARG CONTAINER_USER

# assign default values even when variable exists and is empty
# (ARG default value only applies when variable is non existent)
# + make it persistent at runtime (pass this variable to container)
ENV CONTAINER_UID=${CONTAINER_UID:-0}
ENV CONTAINER_GID=${CONTAINER_GID:-0}
# alpine does not accept users/groups with numerical names
ENV CONTAINER_USER=${CONTAINER_USER:-root}
ENV CONTAINER_GROUP=${CONTAINER_GROUP:-root}

RUN apt-get update && apt-get install -y gcc apt-utils wget procps systemd \
	gettext-base libgmp3-dev libmpfr-dev libmpc-dev apt-utils wget git


RUN python --version


# Install nodejs -----------------------------------------------------------------
RUN wget -qO- https://deb.nodesource.com/setup_23.x | bash -
RUN apt-get install -y nodejs
RUN npm install --global yarn@1.22

# Install Nginx along with the missing dependencies
RUN apt-get install nginx -y --fix-missing

# read config.env in interactive shells & setup for reading it in entrypoint
COPY ./envs/vpn/docker/gui/build_files/bashrc_append /tmp
RUN cat /tmp/bashrc_append | tee -a /root/.bashrc \
        /etc/skel/.bashrc >/dev/null

RUN [ "$CONTAINER_GROUP" != 'root' ] && groupadd -g $CONTAINER_GID $CONTAINER_GROUP || :
RUN [ "$CONTAINER_USER" != 'root' ] && useradd -m -d /home/$CONTAINER_USER \
        -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_USER || :


COPY ./envs/vpn/docker/gui/build_files/entrypoint*.bash /

# Setup subset of fedbiomed code tree
RUN mkdir -p /fedbiomed/envs /data /fbm-node

# Copy files from fedbiomed repo - in the context of fedbiomed top directory
COPY envs/vpn /fedbiomed/envs/vpn
COPY envs/common /fedbiomed/envs/common
COPY envs/common /fedbiomed/envs/common_reference
COPY fedbiomed /fedbiomed/fedbiomed
COPY scripts /fedbiomed/scripts
COPY notebooks /fedbiomed/notebooks
COPY docs/tutorials /fedbiomed/docs/tutorials
COPY fedbiomed_gui /fedbiomed/fedbiomed_gui
COPY pyproject.toml /fedbiomed/pyproject.toml
COPY README.md /fedbiomed/README.md
COPY hatch_build.py /fedbiomed/hatch_build.py

# Alternative option : git clone & strip unused dirs
#  - ensures working with clean repo (docker build begins with a stat(), hangs for
#    a long time if you have zillion files in notebooks/data)
#  - ensures using develop branch (good or bad)

# Copy nginx configuration templates
COPY  ./envs/vpn/docker/gui/build_files/nginx /fedbiomed/nginx

# Prepare for work in context of alternate account
RUN chown -R $CONTAINER_USER:$CONTAINER_GROUP /fbm-node # Fed-BioMed node directory
RUN chown -R $CONTAINER_USER:$CONTAINER_GROUP /fedbiomed
WORKDIR /fedbiomed

# Installs fedbiomed and necessary modules
RUN pip install '.[gui]'

# Changedir to main directory
WORKDIR /

# Create link to access and error log
# RUN ln -sf /dev/stdout /var/log/nginx/access.log && ln -sf /dev/stderr /var/log/nginx/error.log

# transmit build-time values to running container
ENV CONTAINER_BUILD_USER=${CONTAINER_USER}
ENV CONTAINER_BUILD_GROUP=${CONTAINER_GROUP}
ENV CONTAINER_BUILD_UID=${CONTAINER_UID}
ENV CONTAINER_BUILD_GID=${CONTAINER_GID}

# Launch GUI as root to manage identities
USER root:root
ENTRYPOINT ["/entrypoint.bash"]

