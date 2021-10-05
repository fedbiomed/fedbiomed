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


# docker image for mosquitto server
FROM eclipse-mosquitto

RUN apk update && apk add iptables bash procps alpine-sdk linux-headers kmod

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/

COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
