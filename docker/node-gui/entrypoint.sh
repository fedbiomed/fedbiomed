#!/bin/bash
# This script runs as FEDBIOMED_USER after privilege drop
# nginx config was already set up by entrypoint-root.sh

set -e

# Set Gunicorn PORT and HOST
export FBM_GUI_PORT=8000
export FBM_GUI_HOST=localhost

echo "[USER] Starting node-gui as user: $(whoami)"
echo "[USER] SSL activation status: $SSL_ON"

# need an initialized node to start the GUI
if [ ! -f /fbm-node/.fedbiomed -o ! -d /fbm-node/data ] ; then
  echo "Error: no node configuration found in the node base directory"
  exit 1
fi

# Certificates and nginx were already handled by entrypoint-root.sh

# Function to handle shutdown signals
shutdown() {
    echo "[USER] Shutting down gracefully..."
    kill $GUI_PID 2>/dev/null
    exit 0
}

# Trap signals for graceful shutdown
trap shutdown SIGTERM SIGINT

cd /fbm-node

# Start GUI application
echo "[USER] Starting fedbiomed GUI"
fedbiomed node --path /fbm-node gui start --host "$FBM_GUI_HOST" --port "$FBM_GUI_PORT" --production &
GUI_PID=$!

# Wait for the GUI process
wait $GUI_PID