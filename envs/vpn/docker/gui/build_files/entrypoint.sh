#!/bin/bash

# Fed-BioMed - gui container launch script
# - can be launched as unprivileged user account (when it exists)

# caveat: expect `data-folder` to be mounted under same path as in `node` container
# to avoid inconsistencies in dataset declaration
./scripts/fedbiomed_run gui host 0.0.0.0 data-folder /data config config_node.ini start &

# allow to stop/restart the gui without terminating the container
sleep infinity &

wait $!
