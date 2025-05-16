#!/bin/bash

# Fed-BioMed - node container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point

# read functions
source /entrypoint_functions.bash

# read config.env
source ~/bashrc_entrypoint
check_vpn_environ

init_misc_environ
start_wireguard
configure_wireguard
change_path_owner "/fedbiomed" "/fbm-node /home/$CONTAINER_BUILD_USER"

COMMON_DIR="/fedbiomed/envs/common/"
if [ -z "$(ls -1 $COMMON_DIR)" ]; then
  # TODO: test with privilege drop
  # $SETUSER rsync -auxt "/fedbiomed/envs/common_reference/" "$COMMON_DIR"
  rsync -auxt "/fedbiomed/envs/common_reference/" "$COMMON_DIR"
fi


trap finish TERM INT QUIT


# Create node configuration if not existing yet
su -l -c "export FBM_SECURITY_FORCE_SECURE_AGGREGATION=\"${FBM_SECURITY_FORCE_SECURE_AGGREGATION}\" && \
      export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=false && export FBM_RESEARCHER_IP=10.222.0.2 && \
      export FBM_RESEARCHER_PORT=50051 && export PYTHONPATH=/fedbiomed && \
      FBM_SECURITY_TRAINING_PLAN_APPROVAL=\"${FBM_SECURITY_TRAINING_PLAN_APPROVAL:-True}\" \
      FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=\"${FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS:-False}\" \
      fedbiomed component create --component NODE --path /fbm-node --exist-ok" $CONTAINER_USER

# Overwrite node options file if re-launching container
#   eg FBM_NODE_START_OPTIONS="--gpu-only" can be used for starting node forcing GPU usage
su -l -c "echo \"$FBM_NODE_START_OPTIONS\" >/fbm-node/FBM_NODE_START_OPTIONS" $CONTAINER_USER

# Launch node using node options
$SETUSER fedbiomed node start $FBM_NODE_START_OPTIONS &

echo "Node container is ready"
sleep infinity &

wait $!
