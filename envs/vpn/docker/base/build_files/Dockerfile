# Python of the VPN stack. Every image below derives from this one, so this
# single value decides the interpreter of the vpnserver, node and researcher
# containers.
ARG PYTHON_VERSION="3.11"

# temporary builder image for wireguard tools
# - may need update for properly compiling boringtun
# - built on the older suite so its binaries stay compatible with the base image
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

# docker base image for VPN server, Fed-BioMed node and researcher
# - need proper python version for Fed-BioMed
FROM python:${PYTHON_VERSION}-slim-bookworm

RUN apt-get update && apt-get install -y iptables iproute2 iputils-ping bash vim net-tools procps build-essential kmod

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/
