#!/bin/bash

# Fed-BioMed - gui container launch script
# - can be launched as unprivileged user account (when it exists)

# read functions
source /entrypoint_functions.bash

init_misc_environ
change_path_owner "/fedbiomed" "/home/$CONTAINER_BUILD_USER"

# caveat: expect `data-folder` to be mounted under same path as in `node` container
# to avoid inconsistencies in dataset declaration
$SETUSER ./scripts/fedbiomed_run gui --host 0.0.0.0 --production --data-folder /data config config_node.ini start &

# allow to stop/restart the gui without terminating the container
sleep infinity &

wait $!
