# temporary builder image for wireguard tools
# - need coherent system version with base image
# - may need update for properly compiling boringtun
FROM nvidia/cuda:11.5.0-base-ubuntu20.04 as builder

# install wget though nvidia cuda linux gpg key is obsolete
RUN apt-key del 7fa2af80 && apt-get update || true && apt install -y wget
# update nvidia cuda linux gpg repository key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list && \
    sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*

RUN apt-get update && apt-get install -y git build-essential cargo

# install boringtun userspace implementation of wireguard
# 
# match versions of ubuntu & boringtun
#   0.5.2 not yet supported
RUN cargo install --locked --bin boringtun --version ~0.4.0 boringtun

# install needed wireguard-tools
ENV WITH_WGQUICK=yes
RUN git clone https://git.zx2c4.com/wireguard-tools && \
    make -C wireguard-tools/src && \
    make -C wireguard-tools/src install


# docker base image for VPN server, Fed-BioMed node and researcher
# - need proper python version for Fed-BioMed
FROM nvidia/cuda:11.5.0-base-ubuntu20.04

# install wget though nvidia cuda linux gpg key is obsolete
RUN apt-key del 7fa2af80 && apt-get update || true && apt install -y wget
# update nvidia cuda linux gpg repository key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list && \
    sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*

RUN apt-get update && \
    apt-get install -y python3.9-full && \
    apt-get install -y iptables iproute2 iputils-ping bash vim net-tools procps build-essential kmod

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/
