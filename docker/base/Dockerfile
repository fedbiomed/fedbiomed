# Stage 1: Build Node.js frontend
FROM node:20-bullseye AS node-build
WORKDIR /build

# Copy package files for better caching
COPY fedbiomed_gui/ui/package.json fedbiomed_gui/ui/yarn.lock* ./

# Install dependencies
RUN yarn install --frozen-lockfile && \
    yarn cache clean

# Copy UI source and build
COPY fedbiomed_gui/ui/ ./
RUN yarn build

# Stage 2: Build Python package without hatch hook
FROM python:3.10-slim-bullseye AS python-build

ARG FEDBIOMED_GID=1001
ARG FEDBIOMED_UID=1001
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_USER=fedbiomed

ENV FEDBIOMED_GID=${FEDBIOMED_GID} \
    FEDBIOMED_UID=${FEDBIOMED_UID} \
    FEDBIOMED_GROUP=${FEDBIOMED_GROUP} \
    FEDBIOMED_USER=${FEDBIOMED_USER}

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy all required files for pip install
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} pyproject.toml README.md ./
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} fedbiomed ./fedbiomed
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} envs/common ./envs/common
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} envs/vpn ./envs/vpn
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} notebooks ./notebooks
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} docs/tutorials ./docs/tutorials
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} scripts ./scripts
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} hatch_build.py ./hatch_build.py

# Copy fedbiomed_gui structure (needed for pyproject.toml packages)
COPY --chown=${FEDBIOMED_USER}:${FEDBIOMED_GROUP} fedbiomed_gui ./fedbiomed_gui

# Copy pre-built frontend from node-build stage
COPY --from=node-build /build/build ./fedbiomed_gui/ui/build

# Install build tools and build wheel
RUN sed -i '/hatch\.build\.hooks/,/^$/d' pyproject.toml && pip install --no-cache-dir wheel setuptools && \
    pip wheel --no-cache-dir --wheel-dir /wheels '.[researcher, node, gui]'

# Stage 3: Final runtime image
FROM debian:bullseye-slim

ARG FEDBIOMED_GID=1001
ARG FEDBIOMED_UID=1001
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_USER=fedbiomed

ENV FEDBIOMED_GID=${FEDBIOMED_GID} \
    FEDBIOMED_UID=${FEDBIOMED_UID} \
    FEDBIOMED_GROUP=${FEDBIOMED_GROUP} \
    FEDBIOMED_USER=${FEDBIOMED_USER} \
    PATH="/home/${FEDBIOMED_USER}/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl1.1 \
    libffi7 \
    libbz2-1.0 \
    libjpeg62-turbo \
    libpng16-16 \
    liblzma5 \
    libreadline8 \
    libsqlite3-0 \
    zlib1g \
#    python3-tk \
    libgmp10 \
    libmpfr6 \
    libmpc3 \
    gettext-base \
    supervisor \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy Python from official image
COPY --from=python:3.10-slim-bullseye /usr/local/ /usr/local/

# Create user and directories
RUN groupadd --gid $FEDBIOMED_GID $FEDBIOMED_GROUP && \
    useradd -m --shell /bin/bash -d /home/$FEDBIOMED_USER \
    --uid $FEDBIOMED_UID --gid $FEDBIOMED_GID $FEDBIOMED_USER && \
    mkdir -p /home/$FEDBIOMED_USER/fedbiomed && \
    mkdir -p /home/$FEDBIOMED_USER/log && \
    chown -R $FEDBIOMED_USER:$FEDBIOMED_GROUP /home/$FEDBIOMED_USER

# Copy entrypoint
COPY docker/base/entrypoint-base.sh /entrypoint-base.sh
RUN chmod +x /entrypoint-base.sh

WORKDIR /home/${FEDBIOMED_USER}/fedbiomed

# Copy application files with correct ownership
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP README.md ./README.md

USER $FEDBIOMED_USER

# Install fedbiomed from pre-built wheel
COPY --from=python-build --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP /wheels /tmp/wheels
RUN pip install --user --no-cache-dir --no-index --find-links=/tmp/wheels \
    fedbiomed && \
    rm -rf ~/.cache/pip

# Remove fedbiomed packages thaey are no more needd
RUN rm -rf /home/${FEDBIOMED_USER}/fedbiomed_gui/ \ 
    /home/${FEDBIOMED_USER}/fedbiomed \
    /home/${FEDBIOMED_USER}/envs \
    /home/${FEDBIOMED_USER}/notebooks \ 
    /home/${FEDBIOMED_USER}/docs/tutorials 

    
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/entrypoint-base.sh"]