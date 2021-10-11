#!/bin/bash

# Fed-BioMed - restful container launch script
# - launched as root to clean db + get rid of runtime environment variable
# - may drop privileges to CONTAINER_USER at some point

# set identity when we would like to drop privileges
CONTAINER_USER=${CONTAINER_USER:-root}


# we want to start from a fresh config
rm -rf /app/db.sqlite3
#
# glitch : could not have PATH properly set with su -c either via logins.def or .profile/etc.
# and we need more than default PATH
su -c "export PATH=${PATH} ; python manage.py migrate" $CONTAINER_USER
su -c "export PATH=${PATH} ; python manage.py collectstatic --link --noinput" $CONTAINER_USER
su -c "export PATH=${PATH} ; python manage.py createsuperuser --noinput" $CONTAINER_USER

su -c "export PATH=${PATH} ; gunicorn -w 4 -b 0.0.0.0:8000 --log-level debug fedbiomed.wsgi" $CONTAINER_USER &
#sleep infinity &

wait $!