#!/bin/bash

# Fed-BioMed - gui container launch script
# - can be launched as unprivileged user account (when it exists)

export PYTHONPATH=/fedbiomed
eval "$(conda shell.bash hook)"
conda activate fedbiomed-gui
./scripts/fedbiomed_run gui host 0.0.0.0 data-folder /data config config_node.ini start &

sleep infinity &

wait $!
