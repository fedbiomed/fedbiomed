#!/bin/bash

# Fed-BioMed - node container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point

# read functions
source /entrypoint_functions.bash

# read config.env
source ~/bashrc_entrypoint

init_misc_environ

if [ "$USING_NEW_ACCOUNT" ]
then
    # don't use `chown -R` that would cross mountpoints
    find /fedbiomed -mount -exec chown -h $CONTAINER_USER:$CONTAINER_GROUP {} \;
    if [ "$?" -ne 0 ]
    then
        echo "CRITICAL: could not change ownership of /fedbiomed to $CONTAINER_USER:$CONTAINER_GROUP"
        exit 1
    fi
    echo "info: changed ownership of /fedbiomed to $CONTAINER_USER:$CONTAINER_GROUP"

    # far quicker than doing a `find`
    chown -R $CONTAINER_USER:$CONTAINER_GROUP /home/$CONTAINER_BUILD_USER
    if [ "$?" -ne 0 ]
    then
        echo "CRITICAL: could not change ownership of /home/$CONTAINER_BUILD_USER to $CONTAINER_USER:$CONTAINER_GROUP"
        exit 1
    fi
    echo "info: changed ownership of /home/$CONTAINER_BUILD_USER to $CONTAINER_USER:$CONTAINER_GROUP"
    
fi

check_vpn_environ
start_wireguard
configure_wireguard

trap finish TERM INT QUIT

# Cannot launch node at this step because VPN is not yet fully established
# thus it cannot connect to mqtt

sleep infinity &

wait $!
