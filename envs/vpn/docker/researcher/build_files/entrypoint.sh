#!/bin/bash

# Fed-BioMed - researcher container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point

# read config.env
source ~/bashrc_entrypoint

# set identity when we would like to drop privileges
CONTAINER_USER=${CONTAINER_USER:-root}


[ -z "$USE_WG_KERNEL_MOD" ] && USE_WG_KERNEL_MOD=false
RUNNING_KERNELWG=false
RUNNING_BORINGTUN=false

# launch wireguard kernel module or userspace (boringtun)
if "$USE_WG_KERNEL_MOD"
then
    ip link add dev wg0 type wireguard
    [ "$?" -eq 0 ] && RUNNING_KERNELWG=true
fi
# use boringtun if explicitely requested or if kernel wg could not launch properly
if ! "$RUNNING_KERNELWG"
then
    # need to remove kernel module if loaded for using boringtun alternative
    [ -n "$(lsmod | grep wireguard)" ] && rmmod wireguard
    WG_SUDO=1 boringtun wg0 && RUNNING_BORINGTUN=true

fi
# need wireguard to continue
"$RUNNING_KERNELWG" || "$RUNNING_BORINGTUN" || { echo "ERROR: Could not start wireguard" ; exit 1 ; }

CONFIG_DIR=/config
su -c "mkdir -p $CONFIG_DIR/wireguard" $CONTAINER_USER

if [ -s "$CONFIG_DIR/wireguard/wg0.conf" ]
then
    echo "Loading Wireguard config..."
    wg setconf wg0 $CONFIG_DIR/wireguard/wg0.conf
else
    echo "Generating Wireguard config..."
    wg set wg0 private-key <(wg genkey)
    ( umask 0077; wg showconf wg0 | su -c "cat - > $CONFIG_DIR/wireguard/wg0.conf" $CONTAINER_USER )
fi

# VPN client setup
wg set wg0 peer "$VPN_SERVER_PUBLIC_KEY" allowed-ips "$VPN_SERVER_ALLOWED_IPS" endpoint "$VPN_SERVER_ENDPOINT" preshared-key <(echo "$VPN_SERVER_PSK") persistent-keepalive 25
unset VPN_SERVER_PSK

ip -4 address add "$VPN_IP/$VPN_SUBNET_PREFIX" dev wg0
ip link set mtu 1420 up dev wg0
echo "Wireguard started"

finish () {

    echo "Saving Wireguard config"
    ( umask 0077; wg showconf wg0 | su -c "cat - > $CONFIG_DIR/wireguard/wg0.conf" $CONTAINER_USER )

    echo "Stopping Wireguard"
    [ -z "$RUNNING_BORINGTUN" ] && ip link delete dev wg0 || pkill boringtun
    exit 0
}

trap finish TERM INT QUIT

## TODO : make more general by including in the VPN configuration file and in user environment
export MQTT_BROKER=10.220.0.2
export MQTT_BROKER_PORT=1883
export UPLOADS_URL="http://10.220.0.3:8000/upload/"
export PYTHONPATH=/fedbiomed
su -c "export PATH=${PATH} ; eval $(conda shell.bash hook) ; \
    conda activate fedbiomed-researcher ; cd notebooks ; \
    jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' " $CONTAINER_USER &

# proxy port for TensorBoard
# enables launching TB without `--host` option (thus listening only on `localhost`)
# + `watch` for easy respawn in case of failure
    while true ; do \
        socat TCP-LISTEN:6007,fork,reuseaddr,su=$CONTAINER_USER TCP4:127.0.0.1:6006 ; \
        sleep 1 ; \
    done &

sleep infinity &

wait $!
