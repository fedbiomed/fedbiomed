FROM python:3.10-alpine
ENV PYTHONUNBUFFERED 1

# http restful server, expected on port 8844 by fedbiomed
EXPOSE 8000

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


# Copy and install dependencies
COPY requirements.txt /requirements.txt

RUN apk add --no-cache --virtual .build-deps \
    ca-certificates gcc postgresql-dev linux-headers musl-dev \
    libffi-dev jpeg-dev zlib-dev bash \
    && pip install -r requirements.txt

# Creating working directory
RUN mkdir /app
WORKDIR /app

COPY ./entrypoint.bash /

# Expose ports
EXPOSE 8000

ENTRYPOINT ["/entrypoint.bash"]