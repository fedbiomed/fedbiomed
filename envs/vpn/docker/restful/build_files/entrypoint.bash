#!/bin/bash

# Fed-BioMed - restful container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point

# read functions
source /entrypoint_functions.bash

# read config.env
source ~/bashrc_entrypoint

check_vpn_environ
init_misc_environ
start_wireguard
configure_wireguard

trap finish TERM INT QUIT

# we want to start from a fresh config
rm -rf /app/db.sqlite3
#
# glitch : could not have PATH properly set with su -c either via logins.def or .profile/etc.
# and we need more than default PATH
su -c "export PATH=${PATH} ; python manage.py migrate" $CONTAINER_USER
su -c "export PATH=${PATH} ; python manage.py collectstatic --link --noinput" $CONTAINER_USER
su -c "export PATH=${PATH} ; python manage.py createsuperuser --noinput" $CONTAINER_USER

# default timeout 30s is not enough for uploading big models over slow links
su -c "export PATH=${PATH} ; gunicorn -w 4 -b 0.0.0.0:8000 --timeout 900 --log-level debug fedbiomed.wsgi" $CONTAINER_USER &

#sleep infinity &

wait $!
