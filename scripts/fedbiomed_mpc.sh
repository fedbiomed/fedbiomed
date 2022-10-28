#!/bin/bash
###############################################################################################################
### MP-SPDZ Command Execution Interface
###############################################################################################################

# Color configuration -----------------------------------------------------------------------------------------
RED='\033[1;31m' #red
YLW='\033[1;33m' #yellow
GRN='\033[1;32m' #green
NC='\033[0m' #reset
BOLD='\033[1m'

# Base Fed-BioMed directory ------------------------------------------------------------------------------------
[[ -n "$ZSH_NAME" ]] || myname=${BASH_SOURCE[0]}
[[ -n "$ZSH_NAME" ]] && myname=${(%):-%x}
basedir=$(cd $(dirname $myname)/.. || exit ; pwd)

# MP-SPDZ git submodule directory
mpspdz_basedir=$basedir/modules/MP-SPDZ
programs_dir=$mpspdz_basedir/Programs/Source
# ---------------------------------------------------------------------------------------------------------------

# Parsing arguments ---------------------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--compile)
      COMPILE="$2"
      shift # past argument
      shift # past value
      ;;
    -x|--exec)
      EXEC="$2"
      shift # past argument
      shift # past value
      ;;
    -h | --help)
      echo "HELP"
      ;;
    *)
      EXTRA_ARGS+=" $1" # save positional arg
      shift # past argument
      ;;
  esac
done
# ----------------------------------------------------------------------------------------------------------------

# Verif if command is correct -----------------------------------------------------------------------------------------
if [ -n "$COMPILE" ] && [ -n "$EXEC" ]; then
  echo -e "\n${RED}ERROR:${NC}"
  echo -e "${BOLD}Please specify only one of them; '-c | --compile' or '-x | --exec' not both in the same command ${NC}\n"
  exit 1
fi
# -----------------------------------------------------------------------------------------------------------------

# Compile provided MPC script -------------------------------------------------------------------------------------
if [ -n "$COMPILE" ]; then

  # Check MPC script is existing in MPC
  if [ ! -f "$programs_dir/$COMPILE.mpc" ]; then
    echo "$programs_dir/$COMPILE.mpc"
    echo -e "\n${RED}ERROR:${NC}"
    echo -e "${BOLD}Unknown MPC program '$COMPILE'. Please run --help to see available MPC scripts ${NC}\n"
    exit 1
  fi

  # Retrieve bi-prime
  biprime=$(cat $basedir/bin/biprime)

  # Execute compile command
  compile_out=$(cd "$mpspdz_basedir" && python "$COMPILE.mpc" $EXTRA_ARGS --prime="$biprime" 2>&1 )
  if [ ! $? -eq 0 ]; then
    echo -e "\n${RED}ERROR:${NC}"
    echo -e "${BOLD}Above error occurred while compiling MPC program '$COMPILE'.${NC}\n"
    echo -e "##########################################################################"
    echo -e "$compile_out"
    echo -e "##########################################################################\n"
    exit 1
  else
    echo -e "\n${GRN}Compilation is successful!${NC}"
    echo -e "##########################################################################"
    echo -e "$compile_out"
    echo -e "##########################################################################\n"
  fi

fi

# Execute protocol ------------------------------------------------------------------------------------------------
if [ -n "$EXEC" ]; then

  command=$(cd "$mpspdz_basedir" && ./"$EXEC".x $EXTRA_ARGS)

fi
# -----------------------------------------------------------------------------------------------------------------
