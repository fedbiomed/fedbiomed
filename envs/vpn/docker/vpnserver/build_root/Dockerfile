FROM fedbiomed/base

WORKDIR /fedbiomed

ARG CONTAINER_GID
ARG CONTAINER_UID

RUN [ -n "$CONTAINER_GID" ] && groupadd -g $CONTAINER_GID $CONTAINER_GID || :
RUN [ -n "$CONTAINER_UID" ] && useradd -m -d /home/$CONTAINER_UID \
        -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_UID || :

COPY ./vpn vpn/
COPY ./entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]