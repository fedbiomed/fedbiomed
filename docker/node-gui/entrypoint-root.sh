#!/bin/bash
# This script runs as ROOT before privilege drop
# Used for nginx configuration that requires root permissions

set -e

# Set Gunicorn PORT and HOST for nginx upstream
export FBM_GUI_PORT=8000
export FBM_GUI_HOST=localhost

# To avoid envsubst overwriting default nginx variables
export DOLLAR='$'

# Set variable for nginx handling with/without specific domain
if [ -z "$GUI_SERVER_NAME" ] ; then
  export SERVER_NAME_DIRECTIVE=
  export DEFAULT_FAIL=
  export DEFAULT_SUCCEED=' default_server'
else
  export SERVER_NAME_DIRECTIVE="server_name ${GUI_SERVER_NAME};"
  export DEFAULT_FAIL=' default_server'
  export DEFAULT_SUCCEED=
fi

echo "[ROOT] Configuring nginx for node-gui"

# Handle SSL certificates BEFORE generating nginx config
# Check if mounted certs directory exists and has certificates
if [ -d /fbm-node/gui/certs ]; then
    num_files=$(find /fbm-node/gui/certs -mindepth 1 -type f ! -path '*.gitkeep' -printf x 2>/dev/null | wc -c)
    
    if [ "$num_files" != 0 ]; then
        echo "[ROOT] Checking mounted certificates..."
        num_cert=$(find /fbm-node/gui/certs -mindepth 1 -type f -name "*.crt" -printf x 2>/dev/null | wc -c)
        num_key=$(find /fbm-node/gui/certs -mindepth 1 -type f -name "*.key" -printf x 2>/dev/null | wc -c)
        
        if [ "$num_cert" = 1 ] && [ "$num_key" = 1 ]; then
            export SSL_CERTIFICATE=$(find /fbm-node/gui/certs -type f -name "*.crt")
            export SSL_KEY=$(find /fbm-node/gui/certs -type f -name "*.key")
            echo "[ROOT] Found certificates: $SSL_CERTIFICATE and $SSL_KEY"
        else
            echo "[ROOT] ERROR: Invalid certificate files in /fbm-node/gui/certs"
            echo "[ROOT] Expected exactly 1 .crt and 1 .key file, found: $num_cert .crt, $num_key .key"
            exit 1
        fi
    else
        # Generate self-signed certificates
        echo "[ROOT] Generating self-signed SSL certificates"
        export SSL_CERTIFICATE=/fbm-node/gui/certs/fedbiomed-node-gui.crt
        export SSL_KEY=/fbm-node/gui/certs/fedbiomed-node-gui.key
        mkdir -p /fbm-node/gui/certs
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout "$SSL_KEY" \
          -out "$SSL_CERTIFICATE" \
          -subj "/CN=localhost/" 2>/dev/null
        echo "[ROOT] Certificates generated at $SSL_CERTIFICATE"
    fi
else
    # Directory doesn't exist, create it and generate certs
    echo "[ROOT] Creating certs directory and generating certificates"
    export SSL_CERTIFICATE=/fbm-node/gui/certs/fedbiomed-node-gui.crt
    export SSL_KEY=/fbm-node/gui/certs/fedbiomed-node-gui.key
    mkdir -p /fbm-node/gui/certs
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout "$SSL_KEY" \
      -out "$SSL_CERTIFICATE" \
      -subj "/CN=localhost/" 2>/dev/null
    echo "[ROOT] Certificates generated at $SSL_CERTIFICATE"
fi

# Generate nginx config based on SSL setting
if [ "$SSL_ON" = "True" ] || [ "$SSL_ON" = "true" ] || [ "$SSL_ON" = "1" ]; then
  echo "[ROOT] SSL enabled - generating SSL nginx config"
  envsubst < /home/${FEDBIOMED_USER}/nginx/ssl.conf.template > /etc/nginx/conf.d/default.conf
else
  echo "[ROOT] SSL disabled - generating non-SSL nginx config"
  envsubst < /home/${FEDBIOMED_USER}/nginx/no-ssl.conf.template > /etc/nginx/conf.d/default.conf
fi

# Remove default nginx site config
rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
nginx -t || {
  echo "[ROOT] ERROR: nginx configuration test failed"
  cat /var/log/nginx/error.log 2>/dev/null || true
  exit 1
}

# Start nginx (it will drop privileges based on 'user' directive in config)
echo "[ROOT] Starting nginx"
nginx || {
  echo "[ROOT] ERROR: Failed to start nginx"
  cat /var/log/nginx/error.log 2>/dev/null || true
  exit 1
}

echo "[ROOT] nginx started successfully"
