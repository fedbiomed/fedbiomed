# temporary builder image for wireguard tools
# - need coherent system version with mosquitto image
# - may need update for properly compiling boringtun
FROM python:3.10-alpine3.18 as builder

RUN apk update && apk add git alpine-sdk linux-headers cargo

# install boringtun userspace implementation of wireguard
# 
# - match versions of debian & boringtun : up to date 0.5.2 is now ok for this system
RUN cargo install --locked --bin boringtun-cli --version ~0.5.2 boringtun-cli

# install needed wireguard-tools
RUN apk add wireguard-tools

#
# docker image for restful server
#
FROM python:3.10-alpine3.18
ENV PYTHONUNBUFFERED 1

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

# read config.env in interactive shells & setup for reading it in entrypoint
COPY ./bashrc_append /tmp
RUN mkdir /etc/skel
RUN cat /tmp/bashrc_append | tee -a /root/.bashrc /root/bashrc_entrypoint \
        /etc/skel/.bashrc /etc/skel/bashrc_entrypoint >/dev/null

RUN apk update && apk add shadow bash runuser
RUN [ "$CONTAINER_GROUP" != 'root' ] && groupadd -g $CONTAINER_GID $CONTAINER_GROUP || :
RUN [ "$CONTAINER_USER" != 'root' ] && useradd -m -d /home/$CONTAINER_USER \
        -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_USER || :


# Copy and install dependencies
COPY requirements.txt /requirements.txt

# dependencies for wireguard
RUN apk update && apk add iptables bash vim net-tools procps alpine-sdk linux-headers kmod
# dependencies for restful server
RUN apk add --no-cache --virtual .build-deps \
    ca-certificates gcc postgresql-dev linux-headers musl-dev \
    libffi-dev jpeg-dev zlib-dev bash \
    && pip install -r requirements.txt

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun-cli /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/

# Creating working directory
RUN mkdir /app
WORKDIR /app

COPY ./entrypoint*.bash /

# transmit build-time values to running container
ENV CONTAINER_BUILD_USER=${CONTAINER_USER}
ENV CONTAINER_BUILD_GROUP=${CONTAINER_GROUP}
ENV CONTAINER_BUILD_UID=${CONTAINER_UID}
ENV CONTAINER_BUILD_GID=${CONTAINER_GID}

ENTRYPOINT ["/entrypoint.bash"]
