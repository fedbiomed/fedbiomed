#!/bin/bash

source /functions.bash
new_run_time_user

# This functions changes path owner to new container user if it defined in run time
change_path_owner /fedbiomed "/fbm-researcher /home/$FEDBIOMED_USER/log"



## TODO : make more general by including in the VPN configuration file and in user environment
export FBM_SERVER_HOST="${FBM_SERVER_HOST:-localhost}"
export FBM_SERVER_PORT="${FBM_SERVER_PORT:-50051}"

# Force to false for security reason
export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=False

su -c "export FBM_RESEARCHER_COMPONENT_ROOT=/fbm-researcher ; \
	fedbiomed component create -c researcher --path /fbm-researcher  --exist-ok; \
    jupyter notebook /fbm-researcher/notebooks --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' " \
	$CONTAINER_USER &

# proxy port for TensorBoard
# enables launching TB without `--host` option (thus listening only on `localhost`)
# + `watch` for easy respawn in case of failure
    while true ; do \
        socat TCP-LISTEN:6007,fork,reuseaddr,su=$CONTAINER_USER TCP4:127.0.0.1:6006 ; \
        sleep 1 ; \
    done &

echo "Researcher container is ready"
sleep infinity &

wait $!

