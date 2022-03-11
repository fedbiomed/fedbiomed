#!/bin/bash

# Fed-BioMed - gui container launch script
# - can be launched as unprivileged user account (when it exists)

#export PYTHONPATH=/fedbiomed
#eval "$(conda shell.bash hook)"
#conda activate fedbiomed-gui
#./scripts/fedbiomed_run gui data-folder /data config /fedbiomed/etc start

sleep infinity &

wait $!
