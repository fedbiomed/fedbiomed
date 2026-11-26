#!/bin/bash
cat > /home/${FEDBIOMED_USER}/fbm.env <<EOF

FBM_NODE_START_OPTIONS=${FBM_NODE_START_OPTIONS}
FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=${FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS:-False}
FBM_SECURITY_TRAINING_PLAN_APPROVAL=${FBM_SECURITY_TRAINING_PLAN_APPROVAL:-True}
FBM_SECURITY_FORCE_SECURE_AGGREGATION=${FBM_SECURITY_FORCE_SECURE_AGGREGATION:-False}
FBM_SECURITY_SECAGG_INSECURE_VALIDATION=${FBM_SECURITY_SECAGG_INSECURE_VALIDATION:-False}
FBM_RESEARCHER_IP=${FBM_RESEARCHER_IP:-localhost}
FBM_RESEARCHER_PORT=${FBM_RESEARCHER_PORT:-50051}
FBM_SECURITY_ALLOW_FEDERATED_ANALYTICS=${FBM_SECURITY_ALLOW_FEDERATED_ANALYTICS:-False}
FBM_SECURITY_ALLOW_PREPROC=${FBM_SECURITY_ALLOW_PREPROC:-False}

EOF

source /home/${FEDBIOMED_USER}/fbm.env


# Create wrapper script that sources env file
cat > /home/${FEDBIOMED_USER}/start-fbm-node.sh <<'WRAPPER'
#!/bin/bash
set -a  # auto-export all variables
source /home/${FEDBIOMED_USER}/fbm.env
set +a
exec fedbiomed node -p /fbm-node start --background --force ${FBM_NODE_START_OPTIONS}
WRAPPER
chmod +x /home/${FEDBIOMED_USER}/start-fbm-node.sh


# Create node configuration if not existing yet
echo "[INFO] Creating node configuration"
fedbiomed component create --component NODE --path /fbm-node --exist-ok


if [ -n "${MNIST_OFF}" ]; then 
  echo "[INFO] MNIST download and deploy operation is disabled."
elif fedbiomed node -p /fbm-node dataset list | grep -q "'#MNIST'"; then
  echo "[INFO] MNIST dataset already deployed"
else
  echo "[INFO] Deploying MNIST dataset by default"
  fedbiomed node -p /fbm-node dataset add --mnist /fbm-node/data || \
    printf "[ERROR] MNIST download or deploy operation failed. Please make sure you have an internet connection. Otherwise, please set '-e MNIST_OFF=True' in the docker run command.\n"
fi

# Render supervisord config with actual user
echo "Container user: ${FEDBIOMED_USER}"

# Node GUI bind address (GHSA-86g6-36qm-vx82).
# Never bind to all interfaces by default: when the container shares a VPN
# client's network (e.g. `network_mode: service:wg-client`) that would expose
# the GUI on the VPN interface too. By default, bind to the container's own
# Docker-network IP (the IP Docker maps to the container hostname in
# /etc/hosts), which `docker run -p` / compose `ports:` forward to.
# FBM_GUI_HOST overrides it, e.g. 127.0.0.1 (container only) or 0.0.0.0.
if [ -z "${FBM_GUI_HOST}" ]; then
  FBM_GUI_HOST="$(getent ahostsv4 "$(hostname)" | awk 'NR==1{print $1}')"
  case "${FBM_GUI_HOST}" in
    ""|127.*)
      echo "[WARNING] Could not resolve the container IP; Node GUI bound to 127.0.0.1" \
           "(not reachable through published ports). Set FBM_GUI_HOST to change it."
      FBM_GUI_HOST=127.0.0.1
      ;;
  esac
fi
export FBM_GUI_HOST
echo "[INFO] Node GUI listening on ${FBM_GUI_HOST}:8484"
if [ "${FBM_GUI_HOST}" = "0.0.0.0" ]; then
  echo "[WARNING] Node GUI exposed on all interfaces (including VPN interfaces)." \
       "Make sure it is not reachable by untrusted parties."
fi

echo "Creating supervisord conf for management node process"
envsubst '${FEDBIOMED_USER} ${FBM_GUI_HOST}' \
  < /etc/supervisor/supervisord.conf.template > /etc/supervisor/supervisord.conf

# Overwrite node options file if re-launching container
#   eg FBM_NODE_START_OPTIONS="--gpu-only" can be used for starting node forcing GPU usage
echo "$FBM_NODE_START_OPTIONS" >/fbm-node/FBM_NODE_START_OPTIONS

mkdir -p /home/${FEDBIOMED_USER}/log

/home/${FEDBIOMED_USER}/start-fbm-node.sh

# Launch supervisord in foreground to keep container running
exec supervisord -n -c /etc/supervisor/supervisord.conf
