#!/usr/bin/env bash
# Pull repo
git pull

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
    unset FORCE_SCRIPT_NAME
    ;;
  *)
    echo -e "\nUsage: $0 [--build] [--local]"
    exit 1;;
  esac
done

# Start the new one
docker-compose up --force-recreate -d

# Create Admin user
docker exec -it fedbiomed-network sh -c "python manage.py migrate"
docker exec -it fedbiomed-network sh -c "python manage.py collectstatic --link --noinput"
docker exec -it fedbiomed-network sh -c "python manage.py createsuperuser --noinput"
