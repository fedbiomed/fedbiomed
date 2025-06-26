#!/bin/bash

# Fed-BioMed - gui container launch script
# - can be launched as unprivileged user account (when it exists)

# read functions
source /functions.bash
new_run_time_user

# Change path owner to new container user defined in the run time
change_path_owner "/fedbiomed" "/fbm-node /home/$FEDBIOMED_USER"


# To avoid envsubst to over write default nginx variables
export DOLLAR='$'

# Set Gunicorn PORT and HOST
export FBM_GUI_PORT=8000
export FBM_GUI_HOST=localhost

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

echo "SSL activation status:  $SSL_ON"

# Find number of files in /certs directory and ignore .gitkeep
num_files=$(find gui/run_mounts/certs -mindepth 1 -type f ! -path '*.gitkeep' -printf x | wc -c)

if [ "$num_files" != 0 ]; then
     echo "Mounted certs directory is not empty. Checking certificates are existing..."
     num_cert=$(find /certs -mindepth 1 -type f -name "*.crt" -printf x | wc -c)
     num_key=$(find /certs -mindepth 1 -type f -name "*.key" -printf x | wc -c)
     echo "$num_key $num_cert"
     if [ "$num_key" = 0 -a "$num_cert" = 0 ]; then
         echo "ERROR: Mounted directory for certificates is not empty but. There is not file with extension
         'crt' for certificate  and 'key' for the ssl key."
          exit 1
     elif [ "$num_key" = 0 -a "$num_cert" != 0 ] || [ "$num_key" != 0 -a "$num_cert" = 0 ]; then
         echo "Opps something is wrong please make sure that the mounted certs directory contains both crt and key files.
         There is only one of them existing."
         exit 1
     elif [ "$num_key" -gt 1 ] || [ "$num_cert" -gt 1 ]; then
         echo "Please make sure you have only one key and one crt file."
         exit 1
     else
        export SSL_CERTIFICATE=$(find /certs -type f -name "*.crt")
        export SSL_KEY=$(find /certs -type f -name "*.key")
        echo "Found certificates are: $SSL_CERTIFICATE and $SSL_KEY"
     fi
else
    echo "The mounted certificate folder is empty. Generating self-signed certificates."
    export SSL_CERTIFICATE=/certs/fedbiomed-node-gui.crt
    export SSL_KEY=/certs/fedbiomed-node-gui.key

    $SETUSER openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout "$SSL_KEY" \
      -out "$SSL_CERTIFICATE" \
      -subj "/CN=localhost/"
fi

if [ "$SSL_ON" = "True" ] || [ "$SSL_ON" = "true" ] || [ "$SSL_ON" = "1" ]; then
  echo "SSL Has been activated. Remove SSL_ON from environment variable if you want to disable"
  envsubst < /fedbiomed/nginx/ssl.conf.template > /etc/nginx/conf.d/default.conf
else
  echo "SSL is not activated. Set SSL_ON environment variable if you want to enable."
  envsubst < /fedbiomed/nginx/no-ssl.conf.template > /etc/nginx/conf.d/default.conf
fi

cat  /etc/nginx/conf.d/default.conf

if ! rm /etc/nginx/sites-enabled/default; then
  echo "No default configuration found in sites enabled"
fi

if ! service nginx restart; then
  echo "Error while starting nginx server: Here is the error log"
  cat /var/log/nginx/error.log
  exit 1
fi


cd /fbm-node

# need an initialized node to start the GUI
if [ ! -f .fedbiomed -o ! -d ./data ] ; then
  echo "Error: no node configuration found in the node base directory"
  exit 1
fi

# caveat: expect `fbm-node` to be mounted under same path as in `node` container
# to avoid inconsistencies in dataset declaration
$SETUSER fedbiomed node --path /fbm-node gui start --host "$FBM_GUI_HOST" --port "$FBM_GUI_PORT" --production &

# allow to stop/restart the gui without terminating the container
sleep infinity &

wait $!
