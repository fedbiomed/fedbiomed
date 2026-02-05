# !/bin/bash
cat > /home/${FEDBIOMED_USER}/fbm.env <<EOF

FBM_NODE_START_OPTIONS=${FBM_NODE_START_OPTIONS}
FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=${FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS:-True}
FBM_SECURITY_TRAINING_PLAN_APPROVAL=${FBM_SECURITY_TRAINING_PLAN_APPROVAL:-True}
FBM_SECURITY_FORCE_SECURE_AGGREGATION=${FBM_SECURITY_FORCE_SECURE_AGGREGATION:-False}
FBM_SECURITY_SECAGG_INSECURE_VALIDATION=${FBM_SECURITY_SECAGG_INSECURE_VALIDATION:-False}
FBM_RESEARCHER_IP=${FBM_RESEARCHER_IP:-localhost}
FBM_RESEARCHER_PORT=${FBM_RESEARCHER_PORT:-50051}
EOF

source /home/${FEDBIOMED_USER}/fbm.env


# Create wrapper script that sources env file
cat > /home/${FEDBIOMED_USER}/start-fbm-node.sh <<'WRAPPER'
#!/bin/bash
set -a  # auto-export all variables
source /home/${FEDBIOMED_USER}/fbm.env
set +a
exec /home/${FEDBIOMED_USER}/.local/bin/fedbiomed node -p /fbm-node start ${FBM_NODE_START_OPTIONS}
WRAPPER
chmod +x /home/${FEDBIOMED_USER}/start-fbm-node.sh


# Create node configuration if not existing yet
echo "[INFO] Creating node configuration"
fedbiomed component create --component NODE --path /fbm-node --exist-ok


if [ -n "${MNIST_OFF}" ]; then 
  echo "[INFO] MNIST download and deploy operation is disabled."
else
  echo "[INFO] Deploying MNIST dataset by default"
  fedbiomed node -p /fbm-node dataset add --mnist /fbm-node/data || \
    printf "[ERROR] MNIST download or deploy operation failed. Please make sure you have an internet connection. Otherwise, please set '-e MNIST_OFF=True' in the docker run command.\n"
fi

# Render supervisord config with actual user
echo "Container user: ${FEDBIOMED_USER}"

echo "Creating supervisord conf for management node process"
envsubst < /etc/supervisor/supervisord.conf.template > /etc/supervisor/supervisord.conf

# Optional: Check user setup
echo "[ENTRYPOINT] Starting container as user: $CONTAINER_USER"

# Overwrite node options file if re-launching container
#   eg FBM_NODE_START_OPTIONS="--gpu-only" can be used for starting node forcing GPU usage
echo "$FBM_NODE_START_OPTIONS" >/fbm-node/FBM_NODE_START_OPTIONS

# Launch supervisord in foreground to keep container running
exec supervisord -n -c /etc/supervisor/supervisord.conf

