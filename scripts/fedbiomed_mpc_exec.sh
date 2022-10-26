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
# ---------------------------------------------------------------------------------------------------------------

echo "$@"
while [[ $# -gt 0 ]]; do
  case $1 in
    -cf|--compile-file)
      compile_file="$2"
      shift # past argument
      shift # past value
      ;;
    -x|--exec)
      exec_protocol="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=" $1" # save positional arg
      shift # past argument
      ;;
  esac
done