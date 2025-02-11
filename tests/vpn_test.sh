# Colors
RED='\033[0;31m' #red
YLW='\033[1;33m' #yellow
NC='\033[0m' #no color
BLD='\033[1m'

# Error function
function error(){
      echo "${RED}ERROR: ${NC}"
      echo "${BOLD}$1${NC}"
      exit 1
}

function info(){
	  echo "${YLW}INFO:${NC} $1"
}

# Clean images if existing
basedir=$(cd $(dirname $0)/.. || exit ; pwd)
cd $basedir || exit


FEDBIOMED_DIR="$basedir"

info "cleaning images created"

if ! ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean; then
	echo "Error: Can not clean fedbiomed root"
fi

if ! ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean image; then
	echo "Error: Can not clean fedbiomed root"
fi



info "Building VPN Server and Researcher components"

if ! ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn build vpnserver researcher; then
	error "Error while building vpnserver and researcher components"
fi

info "Configuring researcher component"

if ! ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn configure researcher; then
	error "Error while configuring researcher component"
fi


info "Starting researcher"
${FEDBIOMED_DIR}/scripts/fedbiomed_vpn start researcher
if ! docker ps | grep -q fedbiomed-vpn-researcher; then
     error "Fed-BioMed researcher container is not running"
fi


info "Checking status of the researcher VPN connection"
${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status vpnserver researcher


info "Building node and GUI"
if ! ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn build node gui; then
	error "Error while building ndoe and gui images"
fi

cd ${FEDBIOMED_DIR}/envs/vpn/docker
if ! docker compose exec vpnserver bash -ci 'python ./vpn/bin/configure_peer.py genconf node node1'; then
	error "Error while generation configuration file for node 1"
fi

if ! docker compose exec vpnserver bash -ci 'python ./vpn/bin/configure_peer.py genconf node node2'; then
	error "Error while generation configuration file for node 2"
fi



docker compose exec vpnserver cat /config/config_peers/node/node1/config.env > ${FEDBIOMED_DIR}/envs/vpn/docker/node/run_mounts/config/config.env
docker compose exec vpnserver cat /config/config_peers/node/node2/config.env > ${FEDBIOMED_DIR}/envs/vpn/docker/node2/run_mounts/config/config.env


docker compose up -d node
docker compose up -d node2

if ! docker ps | grep -q fedbiomed-vpn-node; then
     error "Fed-BioMed node1 container is not running"
fi

if ! docker ps | grep -q fedbiomed-vpn-node2; then
     error "Fed-BioMed node 2 container is not running"
fi


pbkey_n1="$(docker compose exec node wg show wg0 public-key | tr -d '\r')"
pbkey_n2="$(docker compose exec node2 wg show wg0 public-key | tr -d '\r')"

info "$pbkey_n1"
info "$pbkey_n2"

docker compose exec vpnserver bash -ci "python ./vpn/bin/configure_peer.py add node node1 $pbkey_n1"
docker compose exec vpnserver bash -ci "python ./vpn/bin/configure_peer.py add node node2 $pbkey_n2"


info "Listing registered peers in VPN server"
docker compose exec vpnserver bash -ci "python ./vpn/bin/configure_peer.py list"



if ! ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status node; then
	error "Node is not configured correctly"
fi

if ! ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status node2; then
	error "Node 2 is not configured correctly"
fi

# Disable secure aggregation
export FBM_SECURITY_FORCE_SECURE_AGGREGATION=False


info "Creating a node component in the first node container"
if ! docker compose exec -u $(id -u) node bash -ci 'export FBM_SECURITY_FORCE_SECURE_AGGREGATION='${FBM_SECURITY_FORCE_SECURE_AGGREGATION}'&& export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=false && export FBM_RESEARCHER_IP=10.222.0.2 && export FBM_RESEARCHER_PORT=50051 && export PYTHONPATH=/fedbiomed && FBM_SECURITY_TRAINING_PLAN_APPROVAL=True FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=True fedbiomed component create --component NODE --exist-ok'; then
	error "Can not create node component in node container"
fi

info "Creating a node component in the second node container"
if ! docker compose exec -u $(id -u) node2 bash -ci 'export FBM_SECURITY_FORCE_SECURE_AGGREGATION='${FBM_SECURITY_FORCE_SECURE_AGGREGATION}'&& export FBM_SECURITY_SECAGG_INSECURE_VALIDATION=false && export FBM_RESEARCHER_IP=10.222.0.2 && export FBM_RESEARCHER_PORT=50051 && export PYTHONPATH=/fedbiomed && FBM_SECURITY_TRAINING_PLAN_APPROVAL=True FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=True fedbiomed component create --component NODE --exist-ok'; then
	error "Can not create node component in node container 2"
fi


info "Adding dataset for node 1"
if ! docker compose exec -u $(id -u) node bash -ci 'fedbiomed node dataset add -m /fbm-node/data'; then
	error "Can not add dataset"
fi

docker compose exec -u $(id -u) node bash -ci 'fedbiomed node dataset list'

info "Adding dataset for node 2"
if ! docker compose exec -u $(id -u) node2 bash -ci 'fedbiomed node dataset add -m /fbm-node/data'; then
	error "Can not add dataset"
fi
docker compose exec -u $(id -u) node2 bash -ci 'fedbiomed node dataset list'

info "Starting node 1"
docker compose exec -u $(id -u) node bash -ci "nohup fedbiomed node start >/fbm-node/fedbiomed_node.out &";
if [[ ! $? -eq 0 ]]; then
    error  "Node 1 starting operation failed."
fi


docker compose exec -u $(id -u) node2 bash -ci "nohup fedbiomed node start >/fbm-node/fedbiomed_node.out &";
if [[ ! $? -eq 0 ]]; then
    error  "Node 2 starting operation failed."
fi

sleep 10

docker compose exec -u $(id -u) node bash -ci "cat fbm-node/fedbiomed_node.out"
docker compose exec -u $(id -u) node2 bash -ci "cat fbm-node/fedbiomed_node.out"


info "Convert 101 notebook to python script "
if ! docker compose exec -u $(id -u) researcher bash -ci "jupyter nbconvert /fbm-researcher/notebooks/101_getting-started.ipynb --output=101_getting-started --to script"; then
	error "Error while converting jupyter notebook to python script"
fi

info "start the experiment"
if ! docker compose exec -u $(id -u) researcher bash -ci "python /fbm-researcher/notebooks/101_getting-started.py"; then
	error "Experiment execution has failed!"
fi

info "VPN docker container test is finished!"

info "Cleaning Fed-BioMed docker artifacts"
${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean
${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean image
