# Version tag of Fed-BioMed image
ARG FBM_IMAGE_VERSION

FROM fedbiomed/base:${FBM_IMAGE_VERSION:-latest} AS fedbiomed-base

ARG PYTHON_VERSION=3.11

# Redeclare ARGs from base image to make them available at build time
ARG FEDBIOMED_USER=fedbiomed
ARG FEDBIOMED_GROUP=fedbiomed
ARG FEDBIOMED_UID=2101
ARG FEDBIOMED_GID=2101

# Set environment variables to make them available at runtime
ENV FEDBIOMED_USER=${FEDBIOMED_USER}
ENV FEDBIOMED_GROUP=${FEDBIOMED_GROUP}
ENV FEDBIOMED_UID=${FEDBIOMED_UID}
ENV FEDBIOMED_GID=${FEDBIOMED_GID}

RUN ACTUAL_PYTHON=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') && \
    test "$ACTUAL_PYTHON" = "$PYTHON_VERSION"

# Copy node entry point file
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP docker/node/entrypoint.sh /entrypoint.sh
COPY --chown=$FEDBIOMED_USER:$FEDBIOMED_GROUP docker/node/supervisord-node-api.conf.template /etc/supervisor/supervisord.conf.template

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

USER root:root

# Change ownership of supervisord directories to fedbiomed user
RUN mkdir -p /var/log/supervisor /var/run/supervisor
RUN chown -R ${FEDBIOMED_USER}:${FEDBIOMED_GROUP} /var/log/supervisor /var/run/supervisor && \
    chmod -R 700 /var/log/supervisor /var/run/supervisor

# Allow fedbiomed user to write supervisord config
RUN touch /etc/supervisor/supervisord.conf && \
    chown ${FEDBIOMED_USER}:${FEDBIOMED_GROUP} /etc/supervisor/supervisord.conf

# Create /fbm-node directory with proper ownership
RUN mkdir -p /fbm-node && \
    chown -R ${FEDBIOMED_USER}:${FEDBIOMED_GROUP} /fbm-node

EXPOSE 8484

# Install node-specific dependencies system-wide
RUN pip install --no-cache-dir --no-index \
    --find-links=/tmp/wheels \
    'fedbiomed[node, gui]' && \
    rm -rf /tmp/wheels /root/.cache/pip

# Default to non-root user
USER ${FEDBIOMED_USER}:${FEDBIOMED_GROUP}

WORKDIR /home/${FEDBIOMED_USER}/

# Note: Uses CMD from base image which runs /entrypoint-base.sh
# /entrypoint-base.sh will execute /entrypoint.sh (provided by this image)
