#!/bin/bash

source /functions.bash
new_run_time_user

# This functions changes path owner to new container user if it defined in run time
change_path_owner "/fedbiomed /fbm-researcher /home/$FEDBIOMED_USER/log"

# Server host/port configuration
cat <<EOF > /tmp/fbm-env.sh

export FBM_SERVER_HOST="${FBM_SERVER_HOST:-0.0.0.0}" # Allow local containers to join
export FBM_SERVER_PORT="${FBM_SERVER_PORT:-50051}"

# Force to false for security reason
export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=${FBM_SECURITY_SECAGG_INSECURE_VALIDATION:-False}
export FBM_RESEARCHER_COMPONENT_ROOT=/fbm-researcher
EOF

chmod +x /tmp/fbm-env.sh
source /tmp/fbm-env.sh 

# Function to handle cleanup on exit
cleanup() {
  echo "Shutting down researcher container..."
  kill $(jobs -p) 2>/dev/null
  exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

su -l -c \
  "source /tmp/fbm-env.sh && \
   fedbiomed component create -c researcher --path /fbm-researcher  --exist-ok && \
   echo 'Server port: $FBM_SERVER_PORT' && jupyter notebook /fbm-researcher/notebooks \
    --ip=0.0.0.0 --no-browser --allow-root \
    --NotebookApp.token=''" $CONTAINER_USER &

# proxy port for TensorBoard
# enables launching TB without `--host` option (thus listening only on `localhost`)
# + `watch` for easy respawn in case of failure
while true ; do \
  socat TCP-LISTEN:6007,fork,reuseaddr,su=$CONTAINER_USER TCP4:127.0.0.1:6006 ; \
  sleep 1 ; \
done &

echo "Researcher container is ready"

# Wait for any background job to finish
wait
