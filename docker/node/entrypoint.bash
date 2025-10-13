#!/bin/bash
source /functions.bash

BEFORE_USER_HOOK_SCRIPT="/before-entrypoint-hook.sh"

if [ -x "$BEFORE_USER_HOOK_SCRIPT" ]; then
  echo "[INFO] Executing user hook: $USER_HOOK_SCRIPT"
  source "$BEFORE_USER_HOOK_SCRIPT"
elif [ -f "$BEFORE_USER_HOOK_SCRIPT" ]; then
  echo "[INFO] Found non-executable hook script. Making it executable."
  chmod +x "$BEFORE_USER_HOOK_SCRIPT"
  source "$BEFORE_USER_HOOK_SCRIPT"
else
  echo "[INFO] No user hook script found at $BEFORE_USER_HOOK_SCRIPT, skipping"
fi

new_run_time_user

# This functions changes path owner to new container user if it defined in run time
change_path_owner "/fedbiomed /fbm-node /home/$FEDBIOMED_USER/log"

ls -la /home/$FEDBIOMED_USER

cat <<EOF > /tmp/fbm-env.sh
export FBM_NODE_START_OPTIONS=${FBM_NODE_START_OPTIONS}
export FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=${FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS:-True}
export FBM_SECURITY_TRAINING_PLAN_APPROVAL=${FBM_SECURITY_TRAINING_PLAN_APPROVAL:-True}
export FBM_SECURITY_FORCE_SECURE_AGGREGATION=${FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS:-False}
export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=${FBM_SECURITY_SECAGG_INSECURE_VALIDATION:-False}
export FBM_RESEARCHER_IP=${FBM_RESEARCHER_IP:-localhost}
export FBM_RESEARCHER_PORT=${FBM_RESEARCHER_PORT:-50051}
EOF

chmod +x /tmp/fbm-env.sh
source /tmp/fbm-env.sh

# Create node configuration if not existing yet
echo "[INFO] Creating "
su -l -c "source /tmp/fbm-env.sh && fedbiomed component create --component NODE --path /fbm-node --exist-ok" $CONTAINER_USER


if [ -n "${MNIST_OFF}" ]; then 
  echo "[INFO] MNIST download and deploy operation is disabled."
else
  echo "[INFO] Deploying MNIST dataset by default"
  su -l -c "source /tmp/fbm-env.sh && fedbiomed node -p /fbm-node dataset add --mnist /fbm-node/data" "$CONTAINER_USER" || \
    printf "[ERROR] MNIST download or deploy operation failed. Please make sure you have an internet connection. Otherwise, please set '-e MNIST_OFF=True' in the docker run command.\n"
fi


# Render supervisord config with actual user
export CONTAINER_USER
echo "Container user is assigned as $CONTAINER_USER"

echo "Creating supervisord conf"
envsubst < /etc/supervisor/supervisord.conf.template > /etc/supervisor/supervisord.conf

# Optional: Check user setup
echo "[ENTRYPOINT] Starting container as user: $CONTAINER_USER"

# Overwrite node options file if re-launching container
#   eg FBM_NODE_START_OPTIONS="--gpu-only" can be used for starting node forcing GPU usage
su -l -c "echo \"$FBM_NODE_START_OPTIONS\" >/fbm-node/FBM_NODE_START_OPTIONS" $CONTAINER_USER

# Keep follwing implementation for non su[ervisord case
# su -l -c "source /tmp/fbm-env.sh && fedbiomed node -p /fbm-node start $FBM_NODE_START_OPTIONS &" $CONTAINER_USER

# echo "Node container is ready"
# sleep infinity &

# wait $!

# Drop privileges and launch supervisord
exec su -s /bin/bash "$CONTAINER_USER" -c "supervisord -c /etc/supervisor/supervisord.conf"

