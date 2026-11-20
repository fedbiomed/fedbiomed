ARG PYTHON_VERSION=3.14

# Stage 1: Build Node.js frontend
FROM --platform=$BUILDPLATFORM node:24-bookworm AS node-build
WORKDIR /build

# Copy package files for better caching
COPY packages/node-ui/ui/package.json packages/node-ui/ui/yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile && \
    yarn cache clean

# Copy UI source only. Copying the whole ui/ directory would overlay the host
# node_modules (built for the host OS/arch) on top of the install above.
COPY packages/node-ui/ui/src ./src
COPY packages/node-ui/ui/public ./public
RUN yarn build

# Stage 2: Build Python package without hatch hook
FROM python:${PYTHON_VERSION}-slim-bookworm AS python-build

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
COPY README.md ./README.md
COPY packages/fedbiomed ./packages/fedbiomed
COPY envs/common ./envs/common
COPY notebooks ./notebooks
COPY docs/tutorials ./docs/tutorials
COPY scripts ./scripts

# Copy fedbiomed_gui and fedbiomed_node_api structure (needed for pyproject.toml packages)
COPY packages/node-api ./packages/node-api
COPY packages/node-ui ./packages/node-ui

# Copy pre-built frontend from node-build stage
COPY --from=node-build /build/build ./packages/node-ui/ui/build

# Install build tools and build wheel
ENV FBM_SKIP_FRONTEND_BUILD=1

RUN pip wheel --no-cache-dir \
    --wheel-dir /wheels \
    --find-links=/wheels \
    './packages/fedbiomed[researcher]' \
    ./packages/node-api \
    ./packages/node-ui

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

# Install runtime dependencies the python base image does not already provide.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    libgmp10 \
    libmpfr6 \
    libmpc3 \
    gettext-base \
    supervisor \
    tini \
    && rm -rf /var/lib/apt/lists/*

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
