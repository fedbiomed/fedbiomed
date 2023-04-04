# Notes for Fed-BioMed with VPN

This file explains how to deploy and use Fed-BioMed with VPN encapsulation. Each component is running inside a `docker` container, all communications between Fed-BioMed components pass through the VPN.

## context

Which identity to use ?

- outside of containers : work with the user account used for launching the containers (eg `[user@node $]`), unless this is explicitely written to work as root (eg `[root@node #]`). This is also the account used by the containers when they drop privileges.
- inside containers : follow instructions, work as root for managing vpn (eg `[root@vpnserver-container #]`), work as user for managing experiments (eg `[user@node-container $]`)

Which machine to use ?

- when using distinct machines for the components, type node commands (eg `[user@node $]`) on node, researcher commands (eg `[user@researcher $]`) on researcher, network commands on network (eg `[user@network $]`)
- when running all components on a single laptop, all components are on this machine

## requirements

Supported operating systems for using containers :
  - tested on **Fedora 35**, should work for recent RedHat based Linux
  - tested on **Ubuntu 20.04**, should work for recent Debian based Linux
  - tested on recent **MacOS X**
  - tested on **Windows 10** 21H2 with WSL2 using a Ubuntu-20.04 distribution, should work with most Windows 10/11 and other recent Linux distributions

Pre-requisites for using containers :

* **`docker >= 20.10.0`** is needed to build mqtt, see [there](https://wiki.alpinelinux.org/wiki/Release_Notes_for_Alpine_3.14.0#faccessat2). With older docker version it fails with a `make: sh: Operation not permitted`
* **`docker-compose` >= 1.27.0** is needed for extended file format for [GPU support in docker](https://docs.docker.com/compose/gpu-support/) even if you're not using GPU in container.
  -  some distributions (eg Ubuntu 20.04) don't provide a package with a recent enough version.
  - Type `docker-compose --version` to check installed version.
  - You can use your usual package manager to  install up-to-date version (eg: `sudo apt-get update && sudo apt-get install docker-compose` for apt, `sudo dnf clean metadata && sudo dnf update docker-compose` for dnf).
  - If no suitable package exist for your system, you can use [`docker-compose` install page](https://docs.docker.com/compose/install/).

Installation notes for Windows 10 with WSL2 Ubuntu-20.04:
* build of containers `mqtt` `restful` may fail in `cargo install` step with error `spurious network error [...] Timeout was reached`. This is due to bad name resolution of `crates.io` package respository with default WSL2 DNS configuration. If this happens connect to wsl (`wsl` from Windows command line tool), get admin privileges (`sudo bash`) and create a [`/etc/wsl.conf`](https://docs.microsoft.com/fr-fr/windows/wsl/wsl-config) file containing:
```bash
[network]
generateResolvConf = false
```
* if deploying containers on multiple machines you probably need to [make some ports available](https://docs.microsoft.com/en-us/windows/wsl/networking) on the network (eg: Wireguard server)


## setup VPN and fedbiomed

Tip: build images from a clean file tree (avoid copying modified/config/temporary files to images) :
- method 1 : use a fresh `git clone https://gitlab.inria.fr/fedbiomed/fedbiomed.git` tree
- method 2 : clean your existing file tree
  * general cleaning

  ``` bash
  [user@laptop $] source ./scripts/fedbiomed_environment clean
  ```

  * specific [cleaning](#cleaning) for containers


### (optional) building all images

Optionally build all images in one command. 
Usually build each image separately when initializing each container (see after)

```bash
#[user@laptop $] cd ./envs/vpn/docker
## CONTAINER_{UID,GID,USER,GROUP} are target id for running the container
## **TODO**: check if we can use different id than the account building the images
#
## when running on a single machine : build all needed containers at one time with
#[user@laptop $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g')  docker-compose build base vpnserver mqtt restful basenode node gui researcher
```

### initializing vpnserver

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build base
[user@network $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build vpnserver
```
* set the VPN server public IP *VPN_SERVER_PUBLIC_ADDR*
```bash
[user@network $] cp ./vpnserver/run_mounts/config/config.env.sample ./vpnserver/run_mounts/config/config.env
[user@network $] vi ./vpnserver/run_mounts/config/config.env # change VPN_SERVER_PUBLIC_ADDR
```
* launch container
```bash
[user@network $] docker-compose up -d vpnserver
```
* connect and generate config for components
```bash
[user@network $] docker-compose exec vpnserver bash
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf management mqtt
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf management restful
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf node node1
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf researcher researcher1
```

Run this for all launches of the container :
* launch container
```bash
[user@network $] docker-compose up -d vpnserver
```

### initializing mqtt

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g')  docker-compose build mqtt
```
* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] cp ./vpnserver/run_mounts/config/config_peers/management/mqtt/config.env ./mqtt/run_mounts/config/config.env
```
* launch container
```bash
[user@network $] docker-compose up -d mqtt
```
* retrieve the *publickey*
```bash
[user@network $] docker-compose exec mqtt wg show wg0 public-key
```
* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
[user@network $] docker-compose exec vpnserver python ./vpn/bin/configure_peer.py add management mqtt *publickey*
## other option :
#[user@network $] docker-compose exec vpnserver bash
#[root@vpnserver-container #] python ./vpn/bin/configure_peer.py add management mqtt *publickey*
```
* check the container correctly established a VPN with vpnserver:
```bash
# 10.220.0.1 is vpnserver contacted inside the VPN
# it should answer to the ping
[user@network $] docker-compose exec mqtt ping -c 3 -W 1 10.220.0.1
```

Run this for all launches of the container :
* launch container
```bash
[user@network $] docker-compose up -d mqtt
```

### initializing restful

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g')  docker-compose build restful
```
* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] cp ./vpnserver/run_mounts/config/config_peers/management/restful/config.env ./restful/run_mounts/config/config.env
```
* launch container
```bash
[user@network $] docker-compose up -d restful
```
* retrieve the *publickey*
```bash
[user@network $] docker-compose exec restful wg show wg0 public-key
```
* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
[user@network $] docker-compose exec vpnserver python ./vpn/bin/configure_peer.py add management restful *publickey*
```
* check the container correctly established a VPN with vpnserver:
```bash
# 10.220.0.1 is vpnserver contacted inside the VPN
# it should answer to the ping
[user@network $] docker-compose exec restful ping -c 3 -W 1 10.220.0.1
```

Run this for all launches of the container :
* launch container
```bash
[user@network $] docker-compose up -d restful
```

### initializing node

#### specific instructions: building node image in classical case

This paragraph contains specific instructions for the classical case where
you build the node image on the same machine and in the same code tree where
you run it.

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build basenode
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build node
```
  Alternative: build an (thiner) image without GPU support if you will never use it 
```bash
[user@build $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build basenode-no-gpu
[user@build $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build node
```

Then follow the common instructions for nodes (below).


#### specific instructions: building node image on a different machine

This paragraph contains specific instructions when building node image
on a different machine than the machine where the node runs.

If you do not want to clone the repo and build the node image on a machine, you can
instantiate a node from an image built on another machine.

* in this paragraph we distinguish commands typed on the build machine (eg `[user@build $]`) from the commands typed on the machine running the node (eg `[user@node $]`)


Run this only at first launch of container or after cleaning

On the build machine

* `CONTAINER_UID` and `CONTAINER_GID` used at build time do not need to exist on the build machine, and do not need to be the one used on the node machine for running the node. When launching the container, build-time id is the default id but it may be overloaded. This is useful when building a node image for multiple node machines that don't use the same `CONTAINER_UID` and `CONTAINER_GID`.
* in this example, we build the container with user `fedbiomed` (id `1234`) and group `fedbiomed` (id `1234`). Account name and id used on the node machine may differ (see below).
* build container
```bash
[user@build $] CONTAINER_UID=1234 CONTAINER_GID=1234 CONTAINER_USER=fedbiomed CONTAINER_GROUP=fedbiomed docker-compose build basenode
[user@build $] CONTAINER_UID=1234 CONTAINER_GID=1234 CONTAINER_USER=fedbiomed CONTAINER_GROUP=fedbiomed docker-compose build node
```
  Alternative: build an (thiner) image without GPU support if you will never use it 
```bash
[user@build $] CONTAINER_UID=1234 CONTAINER_GID=1234 CONTAINER_USER=fedbiomed CONTAINER_GROUP=fedbiomed docker-compose build basenode-no-gpu
[user@build $] CONTAINER_UID=1234 CONTAINER_GID=1234 CONTAINER_USER=fedbiomed CONTAINER_GROUP=fedbiomed docker-compose build node
```
* save image for container
```bash
[user@build $] docker image save fedbiomed/vpn-node | gzip >/tmp/vpn-node-image.tar.gz
```
* save files needed for running container
```bash
[user@build $] cd ./envs/vpn/docker
# if needed, clean the configurations in ./node/run_mounts before
[user@build $] tar cvzf /tmp/vpn-node-files.tar.gz ./docker-compose_run_node.yml ./node/run_mounts
```

On the node machine

* load image for container
```bash
[user@node $] docker image load </tmp/vpn-node-image.tar.gz
```
* change to directory you want to use as base directory for running this container
* load files needed for running container
```bash
[user@node $] mkdir -p ./envs/vpn/docker
[user@node $] cd ./envs/vpn/docker
[user@node $] tar xvzf /tmp/vpn-node-files.tar.gz
[user@node $] mv docker-compose_run_node.yml docker-compose.yml
```
* if needed load data to be passed to container
```bash
# example : copy a MNIST dataset
#[user@node $] rsync -auxvt /tmp/MNIST ./node/run_mounts/data/
```

Then follow the common instructions for nodes (below).


#### common instructions: in all cases

Always follow this paragraph for initializing a node, whether you build it on the same machine 
or another machine.

Run this only at first launch of container or after cleaning :

* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
[user@node $] cd ./envs/vpn/docker
[user@node $] vi ./node/run_mounts/config/config.env
# add the content of the command below in the edited file above
[user@network $] cat ./vpnserver/run_mounts/config/config_peers/node/node1/config.env
## if running on a single machine
#[user@laptop $] cp ./vpnserver/run_mounts/config/config_peers/node/node1/config.env ./node/run_mounts/config/config.env
```
* launch container
```bash
[user@node $] NODE=node
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose up -d $NODE
```
Alternative: launch container with Nvidia GPU support activated. Before launching, install [all the pre-requisites for GPU support](#gpu-support-in-container).
```bash
[user@node $] NODE=node-gpu
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose up -d $NODE
```
  * note : `CONTAINER_{UID,GID,USER,GROUP}` are not necessary if using the same identity as in for the build, but they need to have a read/write access to the directories mounted from the host machine's filesystem.
  * note : when using a different identity than at build time, `docker-compose up` may take up to a few dozen seconds to complete and node be ready for using. This is the time for re-assigning some installed resources in the container to the new account.
* retrieve the *publickey*
```bash
[user@node $] docker-compose exec $NODE wg show wg0 public-key
```
* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
[user@network $] docker-compose exec vpnserver python ./vpn/bin/configure_peer.py add node node1 *publickey*
```
* check the container correctly established a VPN with vpnserver:
```bash
# 10.220.0.1 is vpnserver contacted inside the VPN
# it should answer to the ping
[user@node $] docker-compose exec $NODE ping -c 3 -W 1 10.220.0.1
```

Run this for all launches of the container :

* launch container
```bash
# `CONTAINER_{UID,GID,USER,GROUP}` are not needed if they are the same as used for build
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose up -d $NODE
```
* TODO: better package/scripting needed
  Connect again to the node and launch manually, now that the VPN is established
```bash
[user@node $] docker-compose exec -u $(id -u) $NODE bash
# TODO : make more general by including it in the VPN configuration and user environment ?
# TODO : create scripts in VPN environment
# need proper parameters at first launch to create configuration file
[user@node-container $] export MQTT_BROKER=10.220.0.2
[user@node-container $] export MQTT_BROKER_PORT=1883
[user@node-container $] export UPLOADS_URL="http://10.220.0.3:8000/upload/"
[user@node-container $] export PYTHONPATH=/fedbiomed
[user@node-container $] eval "$(conda shell.bash hook)"
[user@node-container $] conda activate fedbiomed-node
# example : add MNIST dataset using persistent (mounted) /data
[user@node-container $] ENABLE_TRAINING_PLAN_APPROVAL=True ALLOW_DEFAULT_TRAINING_PLANS=True python -m fedbiomed.node.cli -am /data
# start the node
# - `--gpu` : default gpu policy == use GPU if available *and* requested by researcher
# - start with training plan approval enabled and default training plans allowed
[user@node-container $] ENABLE_TRAINING_PLAN_APPROVAL=True ALLOW_DEFAULT_TRAINING_PLANS=True python -m fedbiomed.node.cli --start --gpu
# alternative: start the node in background
# [user@node-container $] nohup python -m fedbiomed.node.cli  -s >./fedbiomed_node.out &
```

#### using the node

To add datasets in the node, copy them in the directory `./node/run_mounts/data` on the node host machine :
```bash
[user@node $] ls ./node/run_mounts/data
MNIST
```
- in the node gui (see below), this directory maps to the `/` directory.
- in the node and node gui containers command line, this directory maps to `/data` directory :
```bash
[user@node-container $] ls ./data
MNIST
```

To register approved training plans in the node, copy them in the directory `./node/run_mounts/data` on the node host machine :
```bash
[user@node $] ls ./node/run_mounts/data
my_training_plan.txt
```
- in the node container command line, this directory maps to `/data` directory :
```bash
[user@node-container $] ls ./data
my_training_plan.txt
```
- register a new training plan with :
```bash
[user@node-container $] ./scripts/fedbiomed_run node --register-training-plan
```
- when prompted for the path of the training plan, indicate the `.txt` export of the training plan file (`/data/my_training_plan.txt` in our example)


### initializing node gui (optional)

The node gui is associated with a node, it usually runs on a machine where the node is also installed.

#### specific instructions: building gui image in classical case

This paragraph contains specific instructions for the classical case where
you build the gui image on the same machine and in the same code tree where
you run it.

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build gui
```

#### specific instructions: building gui image on a different machine

This paragraph contains specific instructions when building gui image
on a different machine than the machine where the node and gui run.

If you do not want to clone the repo and build the gui image on a machine, you can
instantiate a gui from an image built on another machine.

* in this paragraph we distinguish commands typed on the build machine (eg `[user@build $]`) from the commands typed on the machine running the node (eg `[user@node $]`)


Run this only at first launch of container or after cleaning

On the build machine

* `CONTAINER_UID` and `CONTAINER_GID` used at build time do not need to exist on the build machine, and do not need to be the one used on the node machine for running the gui. When launching the container, build-time id is the default id but it may be overloaded. This is useful when building a gui image for multiple node machines that don't use the same `CONTAINER_UID` and `CONTAINER_GID`.
* in this example, we build the container with user `fedbiomed` (id `1234`) and group `fedbiomed` (id `1234`). Account name and id used on the node machine may differ (see below).
* build container
```bash
[user@build $] CONTAINER_UID=1234 CONTAINER_GID=1234 CONTAINER_USER=fedbiomed CONTAINER_GROUP=fedbiomed docker-compose build gui
```
* save image for container
```bash
[user@build $] docker image save fedbiomed/vpn-gui | gzip >/tmp/vpn-gui-image.tar.gz
```
* we assume files needed for running container were already installed (see node documentation)

On the node and gui machine

* load image for container
```bash
[user@node $] docker image load </tmp/vpn-gui-image.tar.gz
```
* change to directory you want to use as base directory for running this container
* we assume files and data needed for running container were already installed (see node documentation)

Then follow the common instructions for gui (below).


#### common instructions: in all cases

Always follow this paragraph for initializing a gui, whether you build it on the same machine 
or another machine.

Run this for all launches of the container :

* launch container
```bash
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose up -d gui
```
  * note : `CONTAINER_{UID,GID,USER,GROUP}` are not necessary if using the same identity as in for the build, but they need to have a read/write access to the directories mounted from the node machine's filesystem.
  * note : when using a different identity than at build time, `docker-compose up` may take up to a few dozen seconds to complete and node be ready for using. This is the time for re-assigning some installed resources in the container to the new account.


#### using the gui

Use the node gui from outside the gui container :
* connect to `http://localhost:8484` from your browser.

By default, only connections from `localhost` are authorized. To enable connection to the GUI from any IP address
  - specify the bind IP address at container launch time (eg: your node public IP address `NODE_IP`, or `0.0.0.0` to listen on all node addresses)
```bash
[user@node $] GUI_HOST=0.0.0.0 docker-compose up -d gui
```
  - connect to `http://${NODE_IP}:8484`
  - **warning** allowing connections from non-`localhost` exposes the gui to attacks from the network. Only use with proper third party security measures (web proxy, firewall, etc.) Currently, the provided gui container does not include a user authentication mechanism or encrypted communications for the user.



### initializing researcher

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@researcher $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build base
[user@researcher $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose build researcher
```
* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
[user@researcher $] cd ./envs/vpn/docker
[user@researcher $] vi ./researcher/run_mounts/config/config.env
# add the content of the command below in the edited file above
[user@network $] cat ./vpnserver/run_mounts/config/config_peers/researcher/researcher1/config.env
## if running on a single machine
#[user@laptop $] cp ./vpnserver/run_mounts/config/config_peers/researcher/researcher1/config.env ./researcher/run_mounts/config/config.env
```
* launch container
```bash
[user@researcher $] docker-compose up -d researcher
```
* retrieve the *publickey*
```bash
[user@researcher $] docker-compose exec researcher wg show wg0 public-key
```
* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
[user@network $] docker-compose exec vpnserver python ./vpn/bin/configure_peer.py add researcher researcher1 *publickey*
```
* check the container correctly established a VPN with vpnserver:
```bash
# 10.220.0.1 is vpnserver contacted inside the VPN
# it should answer to the ping
[user@researcher $] docker-compose exec researcher ping -c 3 -W 1 10.220.0.1
```

Run this for all launches of the container :

* TODO: better package/scripting needed
  Connect again to the researcher and launch manually, now that the VPN is established
```bash
[user@researcher $] docker-compose exec -u $(id -u) researcher bash
# TODO : make more general by including it in the VPN configuration and user environment ?
# TODO : create scripts in VPN environment
# need proper parameters at first launch to create configuration file
[user@researcher-container $] export MQTT_BROKER=10.220.0.2
[user@researcher-container $] export MQTT_BROKER_PORT=1883
[user@researcher-container $] export UPLOADS_URL="http://10.220.0.3:8000/upload/"
[user@researcher-container $] export PYTHONPATH=/fedbiomed
[user@researcher-container $] eval "$(conda shell.bash hook)"
[user@researcher-container $] conda activate fedbiomed-researcher
# ... or any other command
[user@researcher-container $] ./notebooks/101_getting-started.py
```

#### using the researcher

Use notebooks from outside the researcher container :
* connect to `http://localhost:8888` from your browser. By default, only connections from `localhost` are authorized

Use tensorboard from outside the researcher container :
* connect to `http://localhost:8888` from your browser and use embedded tensorboard in your notebook as in the `./notebooks/general-tensorboard.ipynb` example :
```python
from fedbiomed.researcher.environ import environ
tensorboard_dir = environ['TENSORBOARD_RESULTS_DIR']
```
```python
%load_ext tensorboard
```
```python
tensorboard --logdir "$tensorboard_dir"
```
* alternatively connect to `http://localhost:6006` from your browser, after starting tensorboard either as an embedded tensorboard in the notebook (see above), or manually in the container with:
```bash
[user@researcher-container $] eval "$(conda shell.bash hook)"
[user@researcher-container $] conda activate fedbiomed-researcher
[user@researcher-container $] tensorboard --logdir runs &
```

To enable connection to the researcher and the tensorboard from any IP address using `RESEARCHER_HOST`
  - specify the bind IP address at container launch time (eg: your server public IP address `SERVER_IP`, or `0.0.0.0` to listen on all server addresses)
```bash
[user@researcher $] RESEARCHER_HOST=${SERVER_IP} docker-compose up -d researcher
```
  - connect to `http://${SERVER_IP}:8888` and `http://${SERVER_IP}:6006`
  - **warning** allowing connections from non-`localhost` exposes the researcher to attacks from the network. Only use with proper third party security measures (web proxy, firewall, etc.) Currently, the provided researcher container does not include a user authentication mechanism or encrypted communications for the user.

To permanently save your notebooks on the researcher host machine (outside of the container) use the directory `./researcher/run_mounts/samples` :
```bash
[user@researcher $] ls ./researcher/run_mounts/samples
my_notebook.ipynb
```
- in the researcher container Jupyter Notebook web GUI this directory maps to the `/samples` directory under the base notebooks dir (which contains Fed-BioMed sample notebooks).
- in the researcher container CLI, this directory maps to `/fedbiomed/notebooks/samples` directory :
```bash
[user@researcher-container $] ls ./notebooks/samples
my_notebook.ipynb
```


## GPU support in container

You can access the host machine GPU accelerator from a node container to speed up training.
- reminder: Fed-BioMed currently [supports only](https://fedbiomed.gitlabpages.inria.fr/user-guide/nodes/using-gpu/) (1) Nvidia GPUs (2) for PyTorch training (3) on node side

Before using a GPU for Fed-BioMed in a `node` docker container, you need to meet the requirements for the host machine:

* a **Nvidia GPU** recent enough (**`Kepler` or newer** generation) to support `450.x` Nvidia drivers that are needed for CUDA 11.x
  Recommended GPU memory is **>= 4GB or more** depending on your training plans size.
* a **supported operating system**
  - tested on **Fedora 35**, should work for recent RedHat based Linux
  - partly tested on **Ubuntu 20.04**, should work for recent Debian based Linux
  - not tested on Windows with WSL2, but should work with Windows 10 version 21H2 or higher, that [support GPU in WSL2](https://docs.microsoft.com/en/windows/wsl/tutorials/gpu-compute)
  - not supported on MacOS (few Nvidia cards, docker virtualized)
* **Nvidia drivers and CUDA >= 11.5.0** (the version used by Fed-BioMed container with GPU support)
* **[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**
* **`docker-compose` version >= 1.27.0** (already installed for container support)


Installation guidelines for requirements:

* Nvidia drivers and CUDA: Type `nvidia-smi` to check driver version installed. You can use your usual package manager (`apt`, `dnf`) or [nvidia CUDA toolkit](https://developer.nvidia.com/cuda-downloads) download. In both cases, commands depend on your machine configuration.
* Nvidia container toolkit: check list of supported systems and [specific instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). First enable the repository, then install the package (eg: `sudo apt-get update && sudo apt-get install nvidia-docker2` on Ubuntu/Debian, `sudo dnf install nvidia-docker2` on CentOS).
  - if your system version is not supported, you can try to use an approaching version. For example, for Fedora 35 we installed and ran the CentOS 8 version with:
```bash
distribution=centos8
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo dnf install nvidia-docker2
```


FAQ for issues with GPU in containers :
* `docker-compose` file format error when launching any container :
```bash
ERROR: The Compose file './docker-compose.yml' is invalid because:
Unsupported config option for services.node-gpu-other: 'runtime'
```
  - you need to update you `docker-compose` version
* `runtime` error when launching `node-gpu` container :
```bash
ERROR: for node-gpu  Cannot create container for service node-gpu:
Unknown runtime specified nvidia
```
  - you need to install Nvidia container toolkit and/or Nvidia drivers and CUDA
* cuda version error when launching `node-gpu` container :
```bash
docker: Error response from daemon: OCI runtime create failed: [...]
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=11.6,
please update your driver to a newer version,  or use an earlier cuda
container: unknown.
```
  - you need to update Nvidia drivers and/or CUDA version on your host machine

## connecting to containers

You can connect to a container only if the corresponding container is already running

* connect on the VPN server / node / mqtt server / restful as root to configure the VPN
```bash
[user@network $] docker-compose exec vpnserver bash
[user@network $] docker-compose exec mqtt bash
[user@network $] docker-compose exec restful bash
[user@node $] docker-compose exec node bash
[user@researcher $] docker-compose exec researcher bash
```
* connect on the node as user to handle experiments
```bash
[user@node $] docker-compose exec -u $(id -u) node bash
[user@node $] docker-compose exec -u $(id -u) gui bash
[user@researcher $] docker-compose exec -u $(id -u) researcher bash
```

Note : can also use commands in the form, so you don't have to be in the docker-compose file directory
```bash
[user@node $] docker container exec -ti -u $(id -u) fedbiomed-vpn-node bash
[user@node $] docker container exec -ti -u $(id -u) fedbiomed-vpn-gui bash
[user@researcher $] docker container exec -ti -u $(id -u) fedbiomed-vpn-researcher bash
```

## cleaning

### vpnserver

```bash
[user@network $] cd ./envs/vpn/docker

# level 1 : container instance
[user@network $] docker-compose rm -sf vpnserver

# level 2 : configuration
[user@network #] rm -rf vpnserver/run_mounts/config/{config.env,config_peers,ip_assign,wireguard}

# level 3 : image
[user@network $] docker image rm fedbiomed/vpn-vpnserver fedbiomed/vpn-base
[user@network $] docker image prune -f
```

### mqtt

```bash
[user@network $] cd ./envs/vpn/docker

# level 1 : container instance
[user@network $] docker-compose rm -sf mqtt

# level 2 : configuration
[user@network $] rm -rf ./mqtt/run_mounts/config/{config.env,wireguard}

# level 3 : image
[user@network $] docker image rm fedbiomed/vpn-mqtt
[user@network $] docker image prune -f
```

### restful

```bash
[user@network $] cd ./envs/vpn/docker

# level 1 : container instance
[user@network $] docker-compose rm -sf restful

# level 2 : configuration
[user@network $] rm -rf ./restful/run_mounts/config/{config.env,wireguard}
[user@network $] rm -rf ./restful/run_mounts/app/data/media/{persistent,uploads}
[user@network $] rm -rf ./restful/run_mounts/app/data/static
[user@network $] rm -f ./restful/run_mounts/app/db.sqlite3
# also clean saved files ? (same for env/developement)

# level 3 : image
[user@network $] docker image rm fedbiomed/vpn-restful
[user@network $] docker image prune -f
```

### node 

```bash
[user@node $] cd ./envs/vpn/docker

# level 1 : container instance
[user@node $] docker-compose rm -sf node

# level 2 : configuration
[user@node $] rm -rf ./node/run_mounts/config/{config.env,wireguard}
[user@node $] rm -rf ./node/run_mounts/{data,envs,etc,var}/*

# level 3 : image
[user@node $] docker image rm fedbiomed/vpn-node fedbiomed/vpn-basenode
[user@network $] docker image prune -f
```

### node gui

```bash
[user@node $] cd ./envs/vpn/docker

# level 1 : container instance
[user@node $] docker-compose rm -sf gui

# level 2 : configuration
[user@node $] rm -rf ./node/run_mounts/{data,etc,var}/*

# level 3 : image
[user@node $] docker image rm fedbiomed/vpn-gui
[user@network $] docker image prune -f
```

### researcher

Same as node

```bash
[user@researcher $] cd ./envs/vpn/docker

# level 1 : container instance
[user@researcher $] docker-compose rm -sf researcher

# level 2 : configuration
[user@researcher $] rm -rf ./researcher/run_mounts/config/{config.env,wireguard}
[user@researcher $] rm -rf ./researcher/run_mounts/{data,etc,samples,runs,var}/*

# level 3 : image
[user@researcher $] docker image rm fedbiomed/vpn-researcher fedbiomed/vpn-base
[user@network $] docker image prune -f
```

## background / wireguard

* boringtun - wireguard userspace rust implementation
    - https://blog.cloudflare.com/boringtun-userspace-wireguard-rust/
    - https://github.com/cloudflare/boringtun

* wireguard tools : https://git.zx2c4.com/wireguard-tools/about/

* wireguard vs openvpn : https://restoreprivacy.com/vpn/wireguard-vs-openvpn/


## managing peers in vpnserver

Peers in VPN server can be listed or removed through `configure_peer.py`. 

**Example:**

Following code snippet will generate configurations for `mqtt` and `restful` component register their public keys in 
VPN server. 
```bash
[user@network $] docker-compose up -d vpnserver
[user@network $] docker-compose exec vpnserver bash
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf management mqtt
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf management restful
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py add management mqtt 1OIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py add management restful 2OIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=
```
After running above commands, there will be two peers registered under the management as `mqtt` and `restful`. These 
peers can be listed with the `list` command.

```bash
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py list
>>> Output:
type        id       prefix         peers
----------  -------  -------------  ------------------------------------------------
management  restful  10.220.0.3/32  ['2OIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=']
management  mqtt     10.220.0.2/32  ['XOIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=']
```
A peer can have multiple registered keys: 

```bash
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py add management mqtt 3OIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py list
>>> Output:
type        id       prefix         peers
----------  -------  -------------  ------------------------------------------------------------------------------------------------
management  restful  10.220.0.3/32  ['2OIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=']
management  mqtt     10.220.0.2/32  ['3OIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=', 'XOIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=']
```

To remove registered keys of the peer from VPN server: 
```bash
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py remove management mqtt
>>> Output:
type        id       prefix         peers
----------  -------  -------------  ------------------------------------------------
management  restful  10.220.0.3/32  ['2OIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=']
management  mqtt     10.220.0.2/32  []
```

`remove` command removes only the registered keys for the given peer. Since the configuration files are not removed from 
the `config_peers` directory, `list` command will still list removed peers. This will allow you to register new key 
for the peer without generating configuration file all over again. 

Config files of peers can be removed with the `removeconf` flag.

```bash
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py removeconf management mqtt
>>> Output:
type        id       prefix         peers
----------  -------  -------------  -----------------------------------------------
management  restful  10.220.0.3/32  ['2OIHVWcDq5+CaDKrQ3G3QAuVnr41ONVFBto1ylBroZg=']
```

## using different identity for build and run

We already documented the use of different values of `CONTAINER_{UID,GID,USER,GROUP}` at build time and at runtime for the `node` and `gui` containers. The build time identity is the default identity at runtime but it can be overloaded when launching the container. This is useful when building a `node` or `gui` image that is used on several node machines that don't use the same identity for running it.

Different values at build time and runtime is also supported by `vpnserver` `mqtt` `restful` and `researcher` containers. Usage is the same as for `node` and `gui`.

Example : build a researcher container with a default user/group `fedbiomed` (id `1234`), run it with the same account as the account on the researcher machine.
```bash
[user@researcher $] CONTAINER_UID=1234 CONTAINER_GID=1234 CONTAINER_USER=fedbiomed CONTAINER_GROUP=fedbiomed docker-compose build researcher
[user@researcher $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un | sed 's/[^[:alnum:]]/_/g') CONTAINER_GROUP=$(id -gn | sed 's/[^[:alnum:]]/_/g') docker-compose up -d researcher
```
