# temporary builder image for wireguard tools
# - need coherent system version with base image
# - may need update for properly compiling boringtun
# - 2023-05: `12.1.1-base-ubuntu22.04` is uptodate version 
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y git build-essential cargo

# install boringtun userspace implementation of wireguard
# 
# - match versions of debian & boringtun : up to date 0.5.2 compiles ok for bullseye, but could not have it working
#   Continue with 0.4.0
RUN cargo install --locked --bin boringtun --version ~0.4.0 boringtun

# install needed wireguard-tools
RUN apt-get install -y wireguard-tools


# docker base image for VPN server, Fed-BioMed node and researcher
# - need proper python version for Fed-BioMed
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.10-full && \
    apt-get install -y iptables iproute2 iputils-ping bash vim net-tools procps build-essential kmod

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/
