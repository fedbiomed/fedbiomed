#!/bin/bash

# Fed-BioMed - functions for container launch script

# minimal checks on VPN environment variables
check_vpn_environ() {
    # check requires VPN variables are set
    local required_vpn_vars=${1:-'VPN_IP VPN_SUBNET_PREFIX VPN_SERVER_ENDPOINT VPN_SERVER_ALLOWED_IPS VPN_SERVER_PUBLIC_KEY VPN_SERVER_PSK'}
    local vars_error=false

    for var in $required_vpn_vars
    do
        value=$(eval "echo \$$var")
        if [ -z "$value" ] 
        then
            vars_error=true
            echo "CRITICAL: required VPN variable $var is not set"  
        else
            [ "$var" = 'VPN_SERVER_PSK' ] && value='_HIDDEN_VALUE_'
            echo "info: set $var='$value'"
        fi
    done
    if "$vars_error"
    then
        echo "CRITICAL: missing some required VPN variables, terminating"
        exit 1
    fi
}

# initialize some environment variables, create new account/group
init_misc_environ() {

    #catch all errors in pipes
    set -o pipefail

    # set base directory for configuration files
    CONFIG_DIR=/config

    # set identity when we would like to drop privileges
    echo "info: container was built with \
CONTAINER_BUILD_USER=${CONTAINER_BUILD_USER} \
CONTAINER_BUILD_UID=${CONTAINER_BUILD_UID} \
CONTAINER_BUILD_GROUP=${CONTAINER_BUILD_GROUP} \
CONTAINER_BUILD_GID=${CONTAINER_BUILD_GID}"
    # should be redundant with bashrc_entrypoint in most cases
    CONTAINER_USER=${CONTAINER_USER:-${CONTAINER_BUILD_USER:-root}}
    CONTAINER_GROUP=${CONTAINER_GROUP:-${CONTAINER_BUILD_GROUP:-root}}
    CONTAINER_UID=${CONTAINER_UID:-${CONTAINER_BUILD_UID:-0}}
    CONTAINER_GID=${CONTAINER_GID:-${CONTAINER_BUILD_GID:-0}}
    echo "info: set CONTAINER_USER=${CONTAINER_USER} \
CONTAINER_UID=${CONTAINER_UID} \
CONTAINER_GROUP=${CONTAINER_GROUP} \
CONTAINER_GID=${CONTAINER_GID}"

    # if using an account/group that does not exist, we need to create it
    # and position a variable to indicate it to the script
    if [ -z "$(getent group $CONTAINER_GID)" -a -z "$(getent group $CONTAINER_GROUP)" ]
    then
        groupadd -g $CONTAINER_GID $CONTAINER_GROUP
        if [ "$?" -ne 0 ]
        then
            echo "CRITICAL: could not create new group CONTAINER_GROUP=$CONTAINER_GROUP CONTAINER_GID=$CONTAINER_GID"
            exit 1
        fi
        echo "info: created new group CONTAINER_GROUP=$CONTAINER_GROUP CONTAINER_GID=$CONTAINER_GID"
        USING_NEW_ACCOUNT=true
    elif [ -z "$(getent group $CONTAINER_GID)" -o -z "$(getent group $CONTAINER_GROUP)" ]
    then
        # case of incoherent group spec : either gid is already used (with another group name)
        # or group name is already used (with another gid)
        echo "CRITICAL: bad group specification, a group already exists with either \
CONTAINER_GROUP=$CONTAINER_GROUP or CONTAINER_GID=$CONTAINER_GID : \
'$(getent group $CONTAINER_GID)' '$(getent group $CONTAINER_GROUP)'"
        exit 1
    fi
    if [ -z "$(getent passwd $CONTAINER_UID)" -a -z "$(getent passwd $CONTAINER_USER)" ]
    then
        # delete stderr to avoid warning messages because homedir already exists
        useradd -m -d /home/$CONTAINER_BUILD_USER \
            -u $CONTAINER_UID -g $CONTAINER_GID -s /bin/bash $CONTAINER_USER 2>/dev/null
        if [ "$?" -ne 0 ]
        then
            echo "CRITICAL: could not create new user CONTAINER_USER=$CONTAINER_USER CONTAINER_UID=$CONTAINER_UID"
            exit 1
        fi
        echo "info: created new user CONTAINER_USER=$CONTAINER_USER CONTAINER_UID=$CONTAINER_UID"
        USING_NEW_ACCOUNT=true
    elif [ -z "$(getent passwd $CONTAINER_UID)" -o -z "$(getent passwd $CONTAINER_USER)" ]
    then
        # case of incoherent user spec : either uid is already used (with another user name)
        # or user name is already used (with another uid)
        echo "CRITICAL: bad user specification, a user already exists with either \
CONTAINER_USER=$CONTAINER_USER or CONTAINER_UID=$CONTAINER_UID : \
'$(getent passwd $CONTAINER_UID)' '$(getent passwd $CONTAINER_USER)'"
        exit 1
    fi

    # set command for dropping privileges
    SETUSER=eval
    [ "$(id -u)" -eq 0 ] && [ ${CONTAINER_USER} != root ] && \
        SETUSER="runuser -u ${CONTAINER_USER} --"

    # do we prefer to use wireguard kernel vs userspace version
    [ -z "$USE_WG_KERNEL_MOD" ] && USE_WG_KERNEL_MOD=false
    echo "info: set USE_WG_KERNEL_MOD='${USE_WG_KERNEL_MOD}'"

    # which version of wireguard is running
    RUNNING_KERNELWG=false
    RUNNING_BORINGTUN=false
}

# change ownership of some directories
change_path_owner() {
    local path_nocross=$1  # path to recursively change, dont cross mount boundaries
    local path_full=$2  # path to recursively change, no additional checks

    if [ "$USING_NEW_ACCOUNT" ]
    then
        for path in $path_nocross
        do
            if [ -e $path ]
            then
                # don't use `chown -R` that would cross mountpoints
                find $path -mount -exec chown -h $CONTAINER_USER:$CONTAINER_GROUP {} \;
                if [ "$?" -ne 0 ]
                then
                    echo "CRITICAL: could not change ownership of $path to $CONTAINER_USER:$CONTAINER_GROUP"
                    exit 1
                fi
                echo "info: changed ownership of $path to $CONTAINER_USER:$CONTAINER_GROUP"
            fi
        done

        for path in $path_full
        do
            if [ -e $path ] 
            then
                # far quicker than doing a `find`
                chown -R $CONTAINER_USER:$CONTAINER_GROUP $path
                if [ "$?" -ne 0 ]
                then
                    echo "CRITICAL: could not change ownership of $path to $CONTAINER_USER:$CONTAINER_GROUP"
                    exit 1
                fi
                echo "info: changed ownership of $path to $CONTAINER_USER:$CONTAINER_GROUP"
            fi
        done
    fi
}

# check wireguard interface exists
check_wg_interface() {

    if ! $(ifconfig wg0 >/dev/null 2>&1)
    then 
        echo "CRITICAL: no wireguard interface wg0, cannot continue"
        exit 1
    fi
}

# launch wireguard VPN
start_wireguard(){

    # launch wireguard kernel module or userspace (boringtun)
    if "$USE_WG_KERNEL_MOD"
    then
        ip link add dev wg0 type wireguard
        [ "$?" -eq 0 ] && RUNNING_KERNELWG=true
    fi

    # use boringtun if explicitly requested or if kernel wg could not launch properly
    if ! "$RUNNING_KERNELWG"
    then
        # need to remove kernel module if loaded for using boringtun alternative
        [ -n "$(lsmod | grep wireguard)" ] && rmmod wireguard
        WG_SUDO=1 boringtun-cli wg0 && RUNNING_BORINGTUN=true
    fi

    # need wireguard to continue
    if "$RUNNING_KERNELWG"
    then
        echo "info: started wireguard, using kernel module"
    elif "$RUNNING_BORINGTUN"
    then
        echo "info: started wireguard, using boringtun"
    else
        echo "CRITICAL: could not start wireguard"
        exit 1
    fi
}

# stop wireguard vpn
stop_wireguard() {

    # check wireguard interface still exists (may be destroyed by some errors)
    check_wg_interface

    echo "info: stopping wireguard"
    if "$RUNNING_BORINGTUN"
    then
        pkill boringtun-cli
        if [ "$?" -ne 0 ]
        then
            echo "CRITICAL: could not stop wireguard boringtun"
            exit 1
        fi
    fi
    if "$RUNNING_KERNELWG"
    then
        rmmod wireguard
        if [ "$?" -ne 0 ]
        then
            echo "CRITICAL: could not stop wireguard kernel module"
            exit 1
        fi
    fi

    [ -z "$RUNNING_BORINGTUN" ] && ip link delete dev wg0 || pkill boringtun-cli

}

save_wg_config() {

    # check wireguard interface still exists (may be destroyed by some errors)
    check_wg_interface

    # save configuration
    ( umask 0077 && wg showconf wg0 | $SETUSER tee $CONFIG_DIR/wireguard/wg0.conf >/dev/null )
    if [ "$?" -ne 0 ]
    then
        echo "CRITICAL: could not save wireguard config to ${CONFIG_DIR}/wireguard/wg0.conf"
        exit 1
    fi 
    echo "info: saved wireguard config to ${CONFIG_DIR}/wireguard/wg0.conf"
}

# configure wireguard VPN
configure_wireguard() {

    $SETUSER mkdir -p $CONFIG_DIR/wireguard
    if [ "$?" -ne 0 ]
    then
        echo "CRITICAL: could not create config directory ${CONFIG_DIR}/wireguard"
        exit 1
    fi

    # need wireguard interface to continue
    check_wg_interface

    if [ -s "$CONFIG_DIR/wireguard/wg0.conf" ]
    then
        echo "info: loading wireguard config from existing file ${CONFIG_DIR}/wireguard/wg0.conf"
        wg setconf wg0 $CONFIG_DIR/wireguard/wg0.conf
        if [ "$?" -ne 0 ]
        then
            echo "CRITICAL: could not set wireguard config from file ${CONFIG_DIR}/wireguard/wg0.conf"
            exit 1
        fi
    else
        echo "info: generating new wireguard config"
        wg set wg0 private-key <(wg genkey)
        if [ "$?" -ne 0 ]
        then
            echo "CRITICAL: could not generate new wireguard config"
            exit 1
        fi  
        
        save_wg_config      
    fi

    # VPN client setup
    if [ -n "$VPN_SERVER_PUBLIC_KEY" ]
    then
        # dont execute on VPN server
        if [ "x$VPN_SERVER_PUBLIC_KEY" != x$(wg show wg0 peers | grep "$VPN_SERVER_PUBLIC_KEY") ]
        then
            # interface may not exist in case of fatal error in previous step
            check_wg_interface

            wg set wg0 peer "$VPN_SERVER_PUBLIC_KEY" allowed-ips "$VPN_SERVER_ALLOWED_IPS" endpoint "$VPN_SERVER_ENDPOINT" preshared-key <(echo "$VPN_SERVER_PSK") persistent-keepalive 25
            if [ "$?" -ne 0 ]
            then 
                echo "CRITICAL: could not add peer to wireguard interface"
                exit 1
            fi

            save_wg_config
        fi
    fi

    # not needed anymore, remove if present (privacy provision)
    unset VPN_SERVER_PSK

    # check interface still exists
    check_wg_interface

    # add address and route
    ip -4 address add "$VPN_IP/$VPN_SUBNET_PREFIX" dev wg0
    if [ "$?" -ne 0 ]
    then 
        echo "CRITICAL: could not set IP address and route for wireguard interface"
        exit 1
    fi

    # activate and set mtu
    ip link set mtu 1420 up dev wg0
    if [ "$?" -ne 0 ]
    then 
        echo "CRITICAL: could not activate wireguard interface"
        exit 1
    fi

    echo "info: wireguard interface ready"
}

# called when stopping the container
finish () {

    echo "info: preparing to stop container"
    save_wg_config
    stop_wireguard
    echo "info: ready to stop container"

    exit 0
}
