#!/usr/bin/env bash
#
# start docker containers for HTTP/MQTT servers
#

# use same id for django container
export CONTAINER_UID=$(id -u)
export CONTAINER_GID=$(id -g) 

# Stop and remove previous containers
docker-compose down && docker-compose rm

export FORCE_SCRIPT_NAME=/fedbiomed
for arg in $*
do
    case $arg in
        --build)
            docker-compose build
            ;;
        --local)
            rm -fr db.sqlite3
            unset FORCE_SCRIPT_NAME
            ;;
        *)
            echo -e "\nUsage: $0 [--build] [--local]"
            exit 1;;
    esac
done
# rq: $0 gives the name of the script file (hence it outputs `deploy.sh`)

# Start the new one
docker-compose up --force-recreate -d

# Create Admin user
docker exec -t fedbiomed-network sh -c "python manage.py migrate"
docker exec -t fedbiomed-network sh -c "python manage.py collectstatic --link --noinput"
docker exec -t fedbiomed-network sh -c "python manage.py createsuperuser --noinput"
