#!/bin/bash
# Server host/port configuration

export FBM_SERVER_HOST="${FBM_SERVER_HOST:-0.0.0.0}" # Allow local containers to join
export FBM_SERVER_PORT="${FBM_SERVER_PORT:-50051}"

# Force to false for security reason
export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=${FBM_SECURITY_SECAGG_INSECURE_VALIDATION:-False}
export FBM_RESEARCHER_COMPONENT_ROOT=/fbm-researcher


# Function to handle cleanup on exit
cleanup() {
  echo "Shutting down researcher container..."
  kill $(jobs -p) 2>/dev/null
  exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

fedbiomed component create -c researcher --path /fbm-researcher  --exist-ok
echo "Starting FBM Jupyter Notebook server..."
echo "Server port: $FBM_SERVER_PORT"
exec jupyter notebook /fbm-researcher/notebooks \
    --ip=0.0.0.0 --no-browser --allow-root \
    --NotebookApp.token=''


# proxy port for TensorBoard
# enables launching TB without `--host` option (thus listening only on `localhost`)
# + `watch` for easy respawn in case of failure
while true ; do \
  socat TCP-LISTEN:6007,fork,reuseaddr,su=$FEDBIOMED_USER TCP4:127.0.0.1:6006 ; \
  sleep 1 ; \
done &

echo "Researcher container is ready"

# Wait for any background job to finish
wait
