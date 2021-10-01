FROM fedbiomed/base

WORKDIR /fedbiomed

ARG CONTAINER_GID
ARG CONTAINER_UID
# assign default values even when variable exists and is empty
# (ARG default value only applies when variable is non existent)
# + make it persistent at runtime (pass this variable to container)
ENV CONTAINER_UID=${CONTAINER_UID:-root}
ENV CONTAINER_GID=${CONTAINER_GID:-root}

RUN [ "$CONTAINER_GID" != 'root' ] && groupadd -g $CONTAINER_GID $CONTAINER_GID || :
RUN [ "$CONTAINER_UID" != 'root' ] && useradd -m -d /home/$CONTAINER_UID \
        -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_UID || :

COPY ./vpn vpn/
COPY ./entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]