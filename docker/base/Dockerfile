ARG PYTHON_VERSION=3.11

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
FROM python:${PYTHON_VERSION}-slim-bookworm AS python-build

ARG FEDBIOMED_GID=2101
ARG FEDBIOMED_UID=2101
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
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} pyproject.toml README.md ./
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} fedbiomed ./fedbiomed
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} envs/common ./envs/common
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} envs/vpn ./envs/vpn
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} notebooks ./notebooks
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} docs/tutorials ./docs/tutorials
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} scripts ./scripts
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} hatch_build.py ./hatch_build.py

# Copy fedbiomed_gui structure (needed for pyproject.toml packages)
COPY --chown=${FEDBIOMED_UID}:${FEDBIOMED_GID} fedbiomed_gui ./fedbiomed_gui

# Copy pre-built frontend from node-build stage
COPY --from=node-build /build/build ./fedbiomed_gui/ui/build

# Install build tools and build wheel
RUN sed -i '/hatch\.build\.hooks/,/^$/d' pyproject.toml && pip install --no-cache-dir wheel setuptools && \
    pip wheel --no-cache-dir --wheel-dir /wheels '.[researcher, node, gui]'

# Stage 3: Final runtime image
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG PYTHON_VERSION

ARG FEDBIOMED_GID=2101
ARG FEDBIOMED_UID=2101
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_USER=fedbiomed

ENV FEDBIOMED_GID=${FEDBIOMED_GID} \
    FEDBIOMED_UID=${FEDBIOMED_UID} \
    FEDBIOMED_GROUP=${FEDBIOMED_GROUP} \
    FEDBIOMED_USER=${FEDBIOMED_USER} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libffi8 \
    libbz2-1.0 \
    libjpeg62-turbo \
    libpng16-16 \
    liblzma5 \
    libreadline8 \
    libsqlite3-0 \
    zlib1g \
    libgmp10 \
    libmpfr6 \
    libmpc3 \
    gettext-base \
    supervisor \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN ACTUAL_PYTHON=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') && \
    test "$ACTUAL_PYTHON" = "$PYTHON_VERSION"

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

# Install fedbiomed system-wide from pre-built wheel
COPY --from=python-build /wheels /tmp/wheels
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels \
    fedbiomed && \
    rm -rf /root/.cache/pip

# Default to non-root user
USER $FEDBIOMED_USER:$FEDBIOMED_GROUP

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/entrypoint-base.sh"]
