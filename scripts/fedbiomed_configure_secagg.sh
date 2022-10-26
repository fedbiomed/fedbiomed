#!/bin/bash

### Configuration


# Color configuration -----------------------------------------------------------------------------------------
RED='\033[1;31m' #red
YLW='\033[1;33m' #yellow
GRN='\033[1;32m' #green
NC='\033[0m' #no color
BOLD='\033[1m'

# Base Fed-BioMed directory ------------------------------------------------------------------------------------
[[ -n "$ZSH_NAME" ]] || myname=${BASH_SOURCE[0]}
[[ -n "$ZSH_NAME" ]] && myname=${(%):-%x}
basedir=$(cd $(dirname $myname)/.. || exit ; pwd)

# MP-SPDZ git submodule directory
mpspdz_basedir=$basedir/modules/MP-SPDZ
# ---------------------------------------------------------------------------------------------------------------


echo -e "\n${GRN}Starting MP-SPDZ configuration...${NC}"
# Clone initialize github submodule if it is not existing
#if [ -z "$(ls -A $mpspdz_basedir)" ]; then
  git submodule update --init modules/MP-SPDZ
#fi

# Get system information  ---------------------------------------------------------------------------------------
echo -e "\n${YLW}--------------------------------SYSTEM INFORMATION------------------------------------------${NC}"
if test $(uname) = "Linux"; then
  echo -e "${BOLD}Linux detected. MP-SPDZ will be used through binary distribution${NC}\n"
  cpu_info='cat /proc/cpuinfo'
elif test $(uname) = "Darwin"; then
  echo -e "${BOLD}macOS detected. MP-SPDZ will be compiled from source instead of using binary distribution${NC}\n"
else
  echo -e "${RED}ERROR${NC}: Unknown operation system. Only Linux or macOS based operating systems are supported\n"
  echo -e "Aborting installation \n"
  exit 1
fi
# ----------------------------------------------------------------------------------------------------------------


# Detect architecture
if test "$cpu_info"; then
  echo -e "${YLW}--------------------------------ARCHITECTURE INFO-------------------------------------------${NC}"
  if $cpu_info | grep -q avx2; then
    echo -e "${BOLD}CPU uses Advanced Vector Extension 2 'avx2'${NC}\n"
    cpu_arch=avx2
  elif $cpu_info | grep -q avx2; then
    cpu_arch=amd64
    echo -e "${BOLD}CPU uses Advanced Micro Devices 64 'amd64'${NC}\n"
  else
    echo -e "${RED}ERROR${NC}: Unknown CPU architecture"
    exit 1
  fi
else
  echo -e "${RED}ERROR${NC}: Can not get CPU info 'cat /proc/cpuinfo' failed!"
  exit 1
fi

# Extract/copy binaries to ${FEDBIOMED_DIR}/bin ------------------------------------------------------------------
echo -e "\n${YLW}Copying binary distributions... ${NC}"
if ! find "$basedir/bin/$(uname)-$cpu_arch/" -name '*.x'  -exec cp -prv '{}' "$basedir/bin/" ';'; then
  echo -e "\n${RED}ERROR${NC}: Can not copy binary files!\n"
  exit 1
fi
# ----------------------------------------------------------------------------------------------------------------


# Locate MPC files ----------------------------------------------------------------------------------------------
echo -e "\n${YLW}Locating MPC files... ${NC}"
if ! find "$basedir/bin/" -name '*.mpc'  -exec cp -prv '{}' "$basedir/modules/MP-SPDZ/Programs/Source/" ';'; then
  echo -e "\n${RED}ERROR${NC}: Can not copy MPC file into MP-SPDZ files!\n"
  exit 1
fi
echo -e "${BOLD}Done! ${NC}"
# ----------------------------------------------------------------------------------------------------------------

# Create Player-Data directory if doesn't exist ------------------------------------------------------------------
echo -e "\n${YLW}Creating Player-Data directory... ${NC}"
if [ ! -d "$basedir/modules/MP-SPDZ/Player-Data" ]; then
  if ! mkdir "$basedir/modules/MP-SPDZ/Player-Data"; then
    echo -e "\n${RED}ERROR${NC}: Can create Player-Data directory!\n"
    exit 1
  fi
fi
echo -e "${BOLD}Done! ${NC}"
# ----------------------------------------------------------------------------------------------------------------


##################################################################################################################
# Create temporary test environment
##################################################################################################################

echo -e "${YLW}\nCreating temporary testing environment----------------------------------------${NC}"

# Create test certificates, input files and ip address -----------------------------------------------------------
echo -e "\n${YLW}Creating temporary certificates and input files for testing${NC}"
player_data="$basedir/modules/MP-SPDZ/Player-Data"

# Remove existing test assigned IP address for each test party
if [ ! -d "$player_data/test_ip_assigned.tldr" ]; then
  rm "$player_data/test_ip_assigned.tldr"
fi

for i in 0 1; do
  openssl req -newkey rsa -nodes -x509 -out "$player_data/P$i.pem" -keyout "$player_data/P$i.key" -subj "/CN=P$i"
  echo "10" > "$player_data/Test-Input-P$i-0"
  echo "localhost:11651$i" >> "$player_data/test_ip_assigned.tldr"
done
echo -e "${BOLD}Done! ${NC}"


echo -e "\n${GRN} MP-SPDZ configuration is successful!\n"

