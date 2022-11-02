#!/bin/bash
###############################################################################################################
### MP-SPDZ Configuration
###############################################################################################################

# Color configuration -----------------------------------------------------------------------------------------
RED='\033[1;31m' #red
YLW='\033[1;33m' #yellow
GRN='\033[1;32m' #green
NC='\033[0m' #no color
BOLD='\033[1m'

# Base Fed-BioMed directory -------------------------------------------------------------------------------------
[[ -n "$ZSH_NAME" ]] || myname=${BASH_SOURCE[0]}
[[ -n "$ZSH_NAME" ]] && myname=${(%):-%x}
basedir=$(cd $(dirname $myname)/.. || exit ; pwd)

# MP-SPDZ git submodule directory
mpspdz_basedir=$basedir/modules/MP-SPDZ
# ---------------------------------------------------------------------------------------------------------------

# Activate conda environment ------------------------------------------------------------------------------------
eval "$(conda shell.bash hook)"
conda activate fedbiomed-node
# ---------------------------------------------------------------------------------------------------------------

echo -e "\n${GRN}Starting MP-SPDZ configuration...${NC}"
# Clone initialize github submodule if it is not existing
echo -e "${BOLD}Updating MP-SPDZ submodule${NC}\n"
git submodule update --init modules/MP-SPDZ

# Get system information  ---------------------------------------------------------------------------------------
echo -e "\n${YLW}--------------------------------SYSTEM INFORMATION------------------------------------------${NC}"
if test $(uname) = "Linux"; then
  echo -e "${BOLD}Linux detected. MP-SPDZ will be used through binary distribution${NC}\n"
  cpu_info='cat /proc/cpuinfo'
elif test $(uname) = "Darwin"; then
  # TODO: Implement macOS installation
  echo -e "${BOLD}macOS detected. MP-SPDZ will be compiled from source instead of using binary distribution${NC}\n"
  echo -e "${YL}\n IMPORTANT: macOS implementation is not completed!"
  exit 1
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

  # Link binaries to ${FEDBIOMED_DIR}/bin ---------------------------------------------------------------------
  echo -e "\n${YLW}Copying binary distributions... ${NC}"
  if ! ln -nsf "$basedir"/bin/$(uname)-$cpu_arch/*.x "$mpspdz_basedir"/; then
    echo -e "\n${RED}ERROR${NC}: Can not copy binary files!\n"
    exit 1
  fi
  # -----------------------------------------------------------------------------------------------------------
else
  echo -e "${RED}ERROR${NC}: Can not get CPU info 'cat /proc/cpuinfo' failed!"
  exit 1
fi

# To use it later
#! find "$basedir/bin/$(uname)-$cpu_arch/" -name '*.x'  -exec cp -prv '{}' "$basedir/bin/" ';'

# Link MPC files ----------------------------------------------------------------------------------------------
# This also includes linking test_setup
echo -e "\n${YLW}Linking MPC files... ${NC}"
if ! ln -nsf "$basedir"/bin/*.mpc "$mpspdz_basedir"/Programs/Source/; then
  echo -e "\n${RED}ERROR${NC}: Can not create link for MPC files into MP-SPDZ programs!\n"
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

rm $mpspdz_basedir/Player-Data/Test-

# Create data for two test party
for i in 0 1 2; do
  openssl req -newkey rsa -nodes -x509 -out "$player_data"/P"$i".pem -keyout "$player_data"/P"$i".key -subj /CN=P"$i"
  echo "10" > "$player_data/Test-Input-P$i-0"
  echo "localhost:11112$i" >> "$player_data/test_ip_assigned.tldr"
done
c_rehash "$mpspdz_basedir"/Player-Data
echo -e "${BOLD}Done! ${NC}"

# Run configuration test-----------------------------------------------------------------------------------------------

# Compiles test setup mpc file
"$basedir"/scripts//fedbiomed_mpc.sh --compile test_setup

# Starts parties for MPC
for i in 0 1 2; do
  "$basedir"/scripts/fedbiomed_mpc.sh --exec shamir-party $i \
      -ip Player-Data/test_ip_assigned.tldr \
      -IF Player-Data/Test-Input \
      -OF Player-Data/Test-Output \
      test_setup \
      -N 3 > /dev/null &
done

# Waits for calculation. There are 3 required output from 3 different party as "RESULT 35"
# when each output is received from parties test will pass. If this process takes mortahn 10 seconds
# test will fail.
count=0
wait=(1 1 1)
while [ $(IFS=+; echo "$((${wait[*]}))") -gt 0 ]; do
  sleep 1;
  for i in 0 1 2; do
    test_result=$(cat "$mpspdz_basedir"/Player-Data/Test-Output-P"$i"-0 2>&1)
    if [ "$test_result" == "RESULT 35" ]; then
        wait[$i]=0
    fi
    count=$((count+1))
  done

  # More than 9 seconds exit process with error
  if [[ "$count" -gt 9 ]]; then
    echo -e "\n${RED}ERROR${NC}: Unknown error occurred while testing MP-SPDZ configuration please check above logs!\n"
    exit 1
  fi
done

echo -e "${BOLD} MP-SPDZ configuration is successfully tested! ${NC}"

# Testing Ends ################################################################################################

echo -e "\n${GRN} MP-SPDZ configuration is successful!\n"
