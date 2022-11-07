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


# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate fedbiomed-researcher

PROTOCOLS=(shamir-party)
SCRIPTS=(server_key test_setup)

function help(){

  case $1 in

    main)

      cat <<EOF
MPC Controller

Usage

> fedbiomed_mpc.sh [compile | exec | shamir-server-key]

compile             : To compile MPC script before execution. Please run 'compile --help' for the usage
                      and more information
exec                : To execute MPC protocol. Please run 'exec --help' for the usage and more information
shamir-server-key   : Executes shamir protocol for server-key computation. Please run 'exec --help' for the usage
                      and more information

EOF
    ;;
    compile)
      cat <<EOF
Compiles MPC scripts

Usage:
> fedbiomed_mpc.sh  compile --script [script-name] [extra arguments]

Script argument is mandatory

-s | --script : The name of the MPC scripts. available scripts -> [test_setup | server_key]

Pass other extra arguments for the compilation. These arguments may very based on MPC script. If the arguments are not
supported compile operation will fail. E.g '--script server_key -N 2'

EOF
    ;;

    exec)
      cat <<EOF

Executes MPC protocols

Usage
> fedbiomed_mpc exec --protocol [protocol-name] [extra-arguments]

Protocol argument is mandatory

-p | --protocol : The name of the MPC protocol. Should be one of available protocols are [shamir-party,]

Extra arguments to pass to the protocol script. The arguments should be supported by the protocol. Otherwise, script
will exit from the execution.


EOF
      ;;
  esac


}

# Compile action ------------------------------------------------------------------------------------------------
function compile(){

  SCRIPT=$1
  EXTRA_ARGS=$2

  # shellcheck disable=SC2076
  if [[ ! " ${SCRIPTS[*]} " =~ " $SCRIPT " ]]; then
    echo -e "\n${RED}ERROR:${NC}"
    echo -e "${BOLD}MPC script '$SCRIPT' is not supported or not existing.${NC}\n"
    exit 1
  fi

  echo -e "${GRN}Compiling '$SCRIPT'...${NC}"
  # Check MPC script is existing in MPC
  if [ ! -f "$mpspdz_basedir/Programs/Source/$SCRIPT.mpc" ]; then
    echo -e "\n${RED}ERROR:${NC}"
    echo -e "${BOLD}Unknown MPC program '$SCRIPT'. Please run --help to see available MPC scripts ${NC}\n"
    exit 1
  fi

  # Retrieve bi-prime
  biprime=$(cat $basedir/bin/biprime)

  # Execute compile command
  compile_out=$(cd "$mpspdz_basedir" && python "Programs/Source/$SCRIPT.mpc" $EXTRA_ARGS --prime="$biprime" 2>&1 )
  if [ ! $? -eq 0 ]; then
    echo -e "\n${RED}ERROR:${NC}"
    echo -e "${BOLD}Above error occurred while compiling MPC program '$SCRIPT'.${NC}\n"
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
}
# ----------------------------------------------------------------------------------------------------------------


function execute_protocol(){

  PROTOCOL=$1
  EXTRA_ARGS=$2


  # shellcheck disable=SC2076
  if [[ ! " ${PROTOCOLS[*]} " =~ " $PROTOCOL " ]]; then
    echo -e "\n${RED}ERROR:${NC}"
    echo -e "${BOLD}The MPC protocol '$PROTOCOL' is not supported or not existing.${NC}\n"
    exit 1
  fi


  echo -e "${GRN}Executing protocol '$PROTOCOL'...${NC}"
  echo -e "${BOLD}Press CTRL+C to stop${NC}"
  exec_out=$(cd "$mpspdz_basedir" && ./"$PROTOCOL".x $EXTRA_ARGS)

  if [ ! $? -eq 0 ]; then
    echo -e "\n${RED}ERROR:${NC}"
    echo -e "${BOLD}Error while executing protocol '$PROTOCOL'. Please check the logs above${NC}\n"
    exit 1
  fi


}


function execute_shamir_server_key(){

  NUMBER_OF_PARTIES=$1
  PARTY_NUMBER=$2
  ASSIGNED_IPS=$3
  OUTPUT_FILE=$4
  COMPILE=$5

  if [ -z "$NUMBER_OF_PARTIES" ] || [ -z "$PARTY_NUMBER" ] || [ -z "$ASSIGNED_IPS" ] || [ -z "$OUTPUT_FILE" ]; then
    echo -e "\n${RED}ERROR:${NC}"
    echo -e "${BOLD}There missing arguments. Please make sure that all the arguments are provided.${NC}\n\
            \rSee 'fedbiomed_mpc shamir-server-key --help'"
    exit 1
  fi


  if [ -n "$COMPILE" ]; then
    if ! "$basedir"/scripts/fedbiomed_mpc.sh compile --script server_key -N "$NUMBER_OF_PARTIES"; then
      echo -e "\n${RED}ERROR:${NC}"
      echo -e "${BOLD}Error while executing protocol '$PROTOCOL'. Please check the logs above${NC}\n"
      exit 1
    fi
  fi

  # Generate the key-share
  if [ ! "$PARTY_NUMBER" -eq 0 ]; then
    echo -e "${GRN}Generating key share for party $PARTY_NUMBER ${NC}"
    if ! python "$basedir"/bin/generate_sk.py --n_len 2048 > "$mpspdz_basedir"/Player-Data/Server-Key-Input-P"$PARTY_NUMBER"-0; then
      echo -e "\n${RED}ERROR:${NC}"
      echo -e "${BOLD}Error while generating key share.${NC}\n"
      exit 1
    fi
    echo -e "${BOLD}Done! Key generated in $mpspdz_basedir/Player-Data/Server-Key-Input-P$PARTY_NUMBER-0${NC}"
  fi
  echo -e "\n${YLW}Execution info:${NC}"
  echo -e "${YLW}---------------------------------------${NC}"
  echo -e "${BOLD}Party no          :${NC} $PARTY_NUMBER"
  echo -e "${BOLD}Number of parties :${NC} $NUMBER_OF_PARTIES"
  echo -e "${BOLD}Assigned IPs      :${NC} $ASSIGNED_IPS"
  echo -e "${BOLD}Output file       :${NC} $OUTPUT_FILE"
  echo -e "${BOLD}Input file        :${NC} $mpspdz_basedir/Player-Data/Server-Key-Input-P$PARTY_NUMBER-0 \n"

  "$basedir"/scripts/fedbiomed_mpc.sh exec --protocol shamir-party "$PARTY_NUMBER" \
        -ip "$ASSIGNED_IPS" \
        -IF "$mpspdz_basedir"/Player-Data/Server-Key-Input \
        -OF "$OUTPUT_FILE" \
        server_key \
        -N "$NUMBER_OF_PARTIES"

}


# Parsing arguments ---------------------------------------------------------------------------------------------
case $1 in

  compile)
    while [[ $# -gt 0 ]]; do
      case $2 in
        -s|--script)
          SCRIPT="$3"
          shift # past argument
          shift # past value
          ;;
        -h| --help)
          help compile
          exit 0
          ;;
        *)
          EXTRA_ARGS+=" $1" # save positional arg
          shift # past argument
          ;;
      esac
    done

    # Compile the protocol
    if [ -n "$SCRIPT" ]; then
          compile "$SCRIPT" "$EXTRA_ARGS"
      else
          echo -e "\n${RED}ERROR:${NC}"
          echo -e "${BOLD}Please specify MPC script name to compile. e.g '--scripts test_setup' ${NC}\n"
          exit 1
    fi
  ;;
  exec)

    # Parse arguments for protocol execution
    while [[ $# -gt 0 ]]; do
      case $2 in
        -s|--protocol)
          PROTOCOL="$3"
          shift # past argument
          shift # past value
          ;;
        -h| --help)
          help exec
          exit 0
          ;;
        *)
          EXTRA_ARGS+=" $2" # save positional arg
          shift # past argument
          ;;
      esac
    done

    # Execute protocol
    if [ -n "$PROTOCOL" ]; then
      execute_protocol "$PROTOCOL" "$EXTRA_ARGS"
    else
       echo -e "\n${RED}ERROR:${NC}"
       echo -e "${BOLD}Please specify protocol name to compile. e.g '--protocol shamir-party' ${NC}\n"
       exit 1
    fi
  ;;

  shamir-server-key)

    # Parse arguments for protocol execution
    while [[ $# -gt 0 ]]; do
      case $2 in
        -nop|--num-of-parties)
          NUMBER_OF_PARTIES="$3"
          shift # past argument
          shift # past value
          ;;
        -aip|--assigned-ips)
          ASSIGNED_IPS="$3"
          shift # past argument
          shift # past value
          ;;
        -pn|--party-number)
          PARTY_NUMBER="$3"
          shift # past argument
          shift # past value
          ;;
       -of|--ouput-file)
          OUTPUT_FILE="$3"
          shift # past argument
          shift # past value
          ;;
        -c|--compile)
          COMPILE=1
          shift # past argument
          shift # past value
          ;;
        *)
          if [ -z  "$2" ]; then
            break;
          else
            echo -e "\n${RED}ERROR:${NC}"
            echo -e "${BOLD}Unknown argument '$2' ${NC}\n"
            exit 1
          fi
          ;;
      esac
    done

    execute_shamir_server_key "$NUMBER_OF_PARTIES" "$PARTY_NUMBER" "$ASSIGNED_IPS" "$OUTPUT_FILE" "$COMPILE"
  ;;
  -h|--help)
    help main
    exit 0
  ;;
  *)
    echo "Please specify the component you want to run between: network, node, researcher, gui"
    exit 1
    ;;
esac

exit 0

