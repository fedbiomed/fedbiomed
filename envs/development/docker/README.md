# Fedbiomed Docker Development Envs

## Fedbiomed Network

This repo provides :

* a RESTful API for uploading models and data associated to fedbiomed
* a MQTT service

These two services are provided as docker containers


## Fedbiomed Jupyter Hub

### Build Containers and run

To build and your jupyterhub containers go to `envs/developments/docker` directory. In docker-compose.yml please specify ip address and ports for accessing MQTT and Restful services;

```
UPLOAD_IP:138.96.220.179
UPLOAD_PORT:8844
UPLOAD_PATH:upload/
MQTT_BROKER:138.96.220.179
MQTT_BROKER_PORT:1883
```

Afterwards, you can run following code to start jupyterhub(make that you are in `envs/developments/docker`)

`docker-compose up --build fed-hub fed-notebook`

### Launching Nodes 
Since jupyter notebook runs in docker, it should download model paramaters using host internal ip. So, node should be started with ip_address option

```
./scripts/fedbiomed_run node config config-n1.ini ip_address x.x.x.x start
```

### Access Jupyter HUB
Open your browser and go http://localhost:8000/fedbiomed. Login with default username and password.

User: fedbiomed
pass: FED123fed

Note: In case of frozen loading bar while accessing jupyterlab notebook, please refresh or logout and login again. 

### Add New User

* add user account to `envs/development/docker/juypterhub/run_mounts/etc/passwd` and `envs/development/docker/juypterhub/run_mounts/etc/shadow` (cut/paste/adapt from existing account)
* generate a password hash (eg `openssl passwd -1`) and add it to `envs/development/docker/juypterhub/run_mounts/etc/shadow`
* commit and push to git to make it permanent
