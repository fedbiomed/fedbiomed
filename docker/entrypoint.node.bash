#!/bin/bash

source /functions.bash
new_run_time_user

# This functions changes path owner to new container user if it defined in run time
change_path_owner /fedbiomed "/fbm-node /data /home/$FEDBIOMED_USER/log"


# Create node configuration if not existing yet
su -l -c "export FBM_SECURITY_FORCE_SECURE_AGGREGATION=\"${FBM_SECURITY_FORCE_SECURE_AGGREGATION}\" && \
      export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=false && export FBM_RESEARCHER_IP=10.222.0.2 && \
      export FBM_RESEARCHER_PORT=50051 && export PYTHONPATH=/fedbiomed && \
      FBM_SECURITY_TRAINING_PLAN_APPROVAL=\"${FBM_SECURITY_TRAINING_PLAN_APPROVAL:-True}\" \
      FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=\"${FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS:-False}\" \
      fedbiomed component create --component NODE --path /fbm-node --exist-ok" $CONTAINER_USER



# Render supervisord config with actual user
echo "Rendering supervisor configuration"
envsubst < /etc/supervisor/supervisord.conf.template > /etc/supervisor/supervisord.conf
echo "Rendering is completed"

# Optional: Check user setup
echo "[ENTRYPOINT] Starting container as user: $CONTAINER_USER"

# Drop privileges and launch supervisord
exec su -s /bin/bash "$CONTAINER_USER" -c "supervisord -c /etc/supervisor/supervisord.conf"

