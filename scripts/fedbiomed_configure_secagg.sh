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

#################################
# Linux Configuration
#################################
configure_linux() {

  cpu_info='cat /proc/cpuinfo'

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

}


#################################
# Darwin Configuration
#################################
configure_darwin() {

  if ! type brew; then
    echo -e "${RED}ERROR:${NC}"
    echo -e "${BOLD} Please install 'Homebrew' to continue configuration"
    exit 1
  fi


  echo -e "\n${YLW}--------------------------------Building from source distribution---------------------------${NC}"

  echo -e "${GRN}\nRunning make clean ...${NC}"
  if ! make -C "$mpspdz_basedir" clean; then
    echo -e "${RED}ERROR:${NC}"
    echo -e "${BOLD}Can not build MP-SPDZ. Please check the logs above"
    exit 1
  fi
  echo -e "${BOLD}Done cleaning! .${NC}"

  echo "MOD = -DGFP_MOD_SZ=33" >> "$mpspdz_basedir"/CONFIG.mine

  echo -e "${GRN}\nInstalling MP-SPDZ from source dist ${NC}"
  if ! make -j 8 -C "$mpspdz_basedir" tldr; then
    echo -e "${RED}ERROR:${NC}"
    echo -e "${BOLD}Can not build MP-SPDZ. Please check the logs above"
    exit 1
  fi
  echo -e "${BOLD}Done make tldr! .${NC}"

  echo -e "${GRN}\nInstalling protocol SHAMIR ${NC}"
  # TODO:
  if ! make -C "$mpspdz_basedir" shamir; then
    echo -e "${RED}ERROR:${NC}"
    echo -e "${BOLD}Can not build shamir protocol. Please check the logs above"
    exit 1
  fi
  echo -e "${BOLD}Done shamir! .${NC}"
}


# Get system information  ---------------------------------------------------------------------------------------
echo -e "\n${YLW}--------------------------------SYSTEM INFORMATION------------------------------------------${NC}"
if test $(uname) = "Linux"; then
  echo -e "${BOLD}Linux detected. MP-SPDZ will be used through binary distribution${NC}\n"
  configure_linux
elif test $(uname) = "Darwin"; then
  echo -e "${BOLD}macOS detected. MP-SPDZ will be compiled from source instead of using binary distribution${NC}\n"
  configure_darwin
else
  echo -e "${RED}ERROR${NC}: Unknown operating system. Only Linux or macOS based operating systems are supported\n"
  echo -e "Aborting installation \n"
  exit 1
fi
# ----------------------------------------------------------------------------------------------------------------


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
if [ -f "$player_data/test_ip_assigned.tldr" ]; then
  rm "$player_data/test_ip_assigned.tldr"
fi

# Remove test input and outputs
if [ -n "$(ls -a "$mpspdz_basedir"/Player-Data/ | grep Test-Output-*)" ]; then
  echo -e "\n${YLW}Removing previous test files${NC}"
  rm "$mpspdz_basedir"/Player-Data/Test-Output*
fi

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
if ! "$basedir"/scripts/fedbiomed_mpc.sh compile --script test_setup; then
    echo -e "\n${RED}ERROR${NC}:"
    echo -e "${BOLD} Error while compiling 'test_setup' ${NC}"
    exit 1
fi

# Starts parties for MPC
for i in 0 1 2; do
  "$basedir"/scripts/fedbiomed_mpc.sh exec --protocol shamir-party $i \
      -ip Player-Data/test_ip_assigned.tldr \
      -IF Player-Data/Test-Input \
      -OF Player-Data/Test-Output \
      test_setup \
      -N 3  &
done

# Waits for calculation. There are 3 required output from 3 different party as "RESULT 35"
# when each output is received from parties test will pass. If this process takes mortahn 10 seconds
# test will fail.
count=0
wait=(1 1 1)
while [ $(IFS=+; echo "$((${wait[*]}))") -gt 0 ]; do
  sleep 1;

  echo -e "${BOLD}Checking the output of parties for testing  round $count out of 9 ------------------------------${NC}"
  for i in 0 1 2; do
    if [ ! -f "$mpspdz_basedir/Player-Data/Test-Output-P$i-0" ]; then
      test_result=''
    else
      test_result=$(cat "$mpspdz_basedir"/Player-Data/Test-Output-P"$i"-0 2>&1)
    fi

    if [ "$test_result" == "RESULT 35" ]; then
        wait[$i]=0
    fi
    echo "Checking output of party ->  $i : Result '$test_result' "
  done
  count=$((count+1))

  # More than 9 seconds exit process with error
  if [[ "$count" -gt 15 ]]; then
    echo -e "\n${RED}ERROR${NC}: Could not verify MP-SPDZ installation expected outputs are not received. \n\
                \r Please check the logs above!\n"
    exit 1
  fi
done

echo -e "*Testings result received at round $(($count-1))"
echo -e "${BOLD}MP-SPDZ configuration is successfully tested! ${NC}"

# Testing Ends ################################################################################################

echo -e "\n${GRN} MP-SPDZ configuration is successful!\n"

