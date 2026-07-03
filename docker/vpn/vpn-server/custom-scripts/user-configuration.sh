#!/bin/bash

# Fed-BioMed - VPN server container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point
source /fedbiomed-entrypoint-base.sh

user_configuration()

# content of VPN environ is not the same for VPN server
# check_vpn_environ 'VPN_IP VPN_SUBNET_PREFIX VPN_MANAGEMENT_IP_ASSIGN VPN_NODE_IP_ASSIGN VPN_RESEARCHER_IP_ASSIGN VPN_SERVER_PUBLIC_ADDR VPN_SERVER_PORT'

