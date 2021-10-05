#!/bin/bash

# Fed-BioMed - django container launch script

python manage.py migrate
python manage.py collectstatic --link --noinput
python manage.py createsuperuser --noinput

gunicorn -w 4 -b 0.0.0.0:8000 --log-level debug fedbiomed.wsgi
