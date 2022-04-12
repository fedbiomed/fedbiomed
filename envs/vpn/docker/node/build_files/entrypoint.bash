#!/bin/bash

# Fed-BioMed - node container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point

# read functions
source /entrypoint_functions.bash

# read config.env
source ~/bashrc_entrypoint



CONTAINER_UID=${CONTAINER_UID:-${CONTAINER_BUILD_UID:-0}}
CONTAINER_GID=${CONTAINER_GID:-${CONTAINER_BUILD_GID:-0}}
CONTAINER_USER=${CONTAINER_USER:-${CONTAINER_BUILD_USER:-root}}
CONTAINER_GROUP=${CONTAINER_GROUP:-${CONTAINER_BUILD_GROUP:-root}}

if [ -z "$(getent group $CONTAINER_GID)" -a -z "$(getent group $CONTAINER_GROUP)" ]
then
    groupadd -g $CONTAINER_GID $CONTAINER_GROUP
    if [ "$?" -ne 0 ]
    then
        echo "CRITICAL: could not create new group CONTAINER_GROUP=$CONTAINER_GROUP CONTAINER_GID=$CONTAINER_GID"
        exit 1
    fi
    echo "info: created new group CONTAINER_GROUP=$CONTAINER_GROUP CONTAINER_GID=$CONTAINER_GID"
    NEW_ACCOUNT=true
fi
if [ -z "$(getent passwd $CONTAINER_UID)" -a -z "$(getent passwd $CONTAINER_USER)" ]
then
    useradd -m -d /home/$CONTAINER_BUILD_USER \
        -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_USER
    if [ "$?" -ne 0 ]
    then
        echo "CRITICAL: could not create new user CONTAINER_USER=$CONTAINER_USER CONTAINER_UID=$CONTAINER_UID"
        exit 1
    fi
    echo "info: created new user CONTAINER_USER=$CONTAINER_USER CONTAINER_UID=$CONTAINER_UID"
    NEW_ACCOUNT=true
fi

if [ "$NEW_ACCOUNT" ]
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
init_misc_environ
start_wireguard
configure_wireguard

trap finish TERM INT QUIT

# Cannot launch node at this step because VPN is not yet fully established
# thus it cannot connect to mqtt

sleep infinity &

wait $!
