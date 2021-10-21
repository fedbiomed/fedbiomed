# temporary builder image for wireguard tools
# - need coherent system version with mosquitto image
# - may need update for properly compiling boringtun (eg: cargo build of `clap` package failing on `buster-slim`)
FROM alpine:3.14 as builder

RUN apk update && apk add git alpine-sdk linux-headers cargo

# install boringtun userspace implementation of wireguard
RUN cargo install --bin boringtun boringtun

# install needed wireguard-tools
ENV WITH_WGQUICK=yes
RUN git clone https://git.zx2c4.com/wireguard-tools && \
    make -C wireguard-tools/src && \
    make -C wireguard-tools/src install

#
# docker image for mosquitto server
#
FROM eclipse-mosquitto

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

RUN apk update && apk add shadow bash
RUN [ "$CONTAINER_GROUP" != 'root' ] && groupadd -g $CONTAINER_GID $CONTAINER_GROUP || :
RUN [ "$CONTAINER_USER" != 'root' ] && useradd -m -d /home/$CONTAINER_USER \
        -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_USER || :


RUN apk update && apk add iptables bash vim net-tools procps alpine-sdk linux-headers kmod

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/

COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
