# temporary builder image for wireguard tools
# - need coherent system version with mosquitto image
# - may need update for properly compiling boringtun
# - 2023-05: `3.18.0` is uptodate version
FROM alpine:3.18.0 as builder

RUN apk update && apk add git alpine-sdk linux-headers cargo

# install boringtun userspace implementation of wireguard
# 
# - match versions of debian & boringtun : up to date 0.5.2 compiles ok for bullseye, but could not have it working
#   Continue with 0.4.0
RUN cargo install --locked --bin boringtun --version ~0.4.0 boringtun

RUN apk add wireguard-tools

#
# docker image for mosquitto server
#
FROM eclipse-mosquitto:2.0.15

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

RUN apk update && apk add shadow bash
RUN [ "$CONTAINER_GROUP" != 'root' ] && groupadd -g $CONTAINER_GID $CONTAINER_GROUP || :
RUN [ "$CONTAINER_USER" != 'root' ] && useradd -m -d /home/$CONTAINER_USER \
        -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_USER || :


RUN apk update && apk add iptables bash runuser vim net-tools procps alpine-sdk linux-headers kmod

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/

COPY ./entrypoint*.bash /

# transmit build-time values to running container
ENV CONTAINER_BUILD_USER=${CONTAINER_USER}
ENV CONTAINER_BUILD_GROUP=${CONTAINER_GROUP}
ENV CONTAINER_BUILD_UID=${CONTAINER_UID}
ENV CONTAINER_BUILD_GID=${CONTAINER_GID}

ENTRYPOINT ["/entrypoint.bash"]
