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
change_path_owner "/fbm-node" "/fedbiomed" "/home/$CONTAINER_BUILD_USER"
start_wireguard
configure_wireguard

COMMON_DIR="/fedbiomed/envs/common/"
if [ -z "$(ls -1 $COMMON_DIR)" ]; then
  # TODO: test with privilege drop
  # $SETUSER rsync -auxt "/fedbiomed/envs/common_reference/" "$COMMON_DIR"
  rsync -auxt "/fedbiomed/envs/common_reference/" "$COMMON_DIR"
fi


trap finish TERM INT QUIT


# Create node configuration if not existing yet
su -c "export FBM_SECURITY_FORCE_SECURE_AGGREGATION=\"${FBM_SECURITY_FORCE_SECURE_AGGREGATION}\" && \
      export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=false && export FBM_RESEARCHER_IP=10.222.0.2 && \
      export FBM_RESEARCHER_PORT=50051 && export PYTHONPATH=/fedbiomed && \
      FBM_SECURITY_TRAINING_PLAN_APPROVAL=True FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=True \
      fedbiomed component create --component NODE --exist-ok" $CONTAINER_USER

# Overwrite node options file if re-launching container
#   eg FBM_NODE_OPTIONS="--gpu-only" can be used for starting node forcing GPU usage
su -c "echo \"$FBM_NODE_OPTIONS\" >/fbm-node/FBM_NODE_OPTIONS" $CONTAINER_USER

# Launch node using node options
$SETUSER fedbiomed node start $FBM_NODE_OPTIONS &


sleep infinity &

wait $!
