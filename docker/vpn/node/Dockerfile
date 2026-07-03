ARG FBM_CONTAINER_VERSION_TAG

# temporary builder image for wireguard tools
# - need coherent system version with base image
# - may need update for properly compiling boringtun
# - 2023-05: `bookworm` is not officially released, stick to `bullseye` for now
FROM debian:bullseye-slim AS builder

RUN apt-get update && apt-get install -y git build-essential curl

# install boringtun userspace implementation of wireguard
# 
# - match versions of debian & boringtun : up to date 0.5.2 compiles ok for bullseye, but could not have it working
#   Continue with 0.4.0
# - glitch: bullseye's apt-get cargo is too old vs boringtun's cargo packages dependencies, need to install
#   up to date rust/cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && . /root/.cargo/env && \
    cargo install --locked --bin boringtun --version ~0.4.0 boringtun

# install needed wireguard-tools
RUN apt-get install -y wireguard-tools

FROM fedbiomed/node:${FBM_CONTAINER_VERSION_TAG:-latest}

ARG PYTHON_VERSION="3.11"

# Redeclare ARGs from base image to make them available at build time
ARG FEDBIOMED_USER=fedbiomed
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_UID=2101
ARG FEDBIOMED_GID=2101

# Set environment variables to make them available at runtime
ENV FEDBIOMED_USER=${FEDBIOMED_USER}
ENV FEDBIOMED_GROUP=${FEDBIOMED_GROUP}
ENV FEDBIOMED_UID=${FEDBIOMED_UID}
ENV FEDBIOMED_GID=${FEDBIOMED_GID}

USER root:root

RUN apt-get update && apt-get install -y iptables iproute2 iputils-ping bash vim net-tools procps build-essential kmod

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/


# This will be executed by VPN node entrypoint
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} docker/vpn/scripts/entrypoint-functions.sh /entrypoint-functions.sh
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} docker/vpn/node/entrypoint-root.sh /entrypoint-root.sh
