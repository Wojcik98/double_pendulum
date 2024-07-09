#!/bin/bash

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#==
# Functions
#==

# Make sure processes in the container can connect to the x server
xauth_stuff() {
    XAUTH=/tmp/.docker.xauth
    if [ ! -f $XAUTH ]
    then
        xauth_list=$(xauth nlist $DISPLAY)
        xauth_list=$(sed -e 's/^..../ffff/' <<< "$xauth_list")
        if [ ! -z "$xauth_list" ]
        then
            echo "$xauth_list" | xauth -f $XAUTH nmerge -
        else
            touch $XAUTH
        fi
        chmod a+r $XAUTH
    fi
}

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [run] [start] [stop] -- Utility for handling docker in AI Olympics."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help         Display the help content."
    echo -e "\tstart              Build the docker image and create the container in detached mode."
    echo -e "\tenter              Begin a new bash process within an existing ai_olympics container."
    echo -e "\tcopy               Copy build and logs artifacts from the container to the host machine."
    echo -e "\tstop               Stop the docker container and remove it."
    echo -e "\tpush               Push the docker image to the cluster."
    echo -e "\tjob                Submit a job to the cluster."
    echo -e "\n" >&2
}

install_apptainer() {
    # Installation procedure from here: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages
    read -p "[INFO] Required 'apptainer' package could not be found. Would you like to install it via apt? (y/N)" app_answer
    if [ "$app_answer" != "${app_answer#[Yy]}" ]; then
        sudo apt update && sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:apptainer/ppa
        sudo apt update && sudo apt install -y apptainer
    else
        echo "[INFO] Exiting because apptainer was not installed"
        exit
    fi
}


#==
# Main
#==

# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 1
fi

# check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
    exit 1
fi

# parse arguments
mode="$1"
# resolve mode
case $mode in
    start)
        xauth_stuff
        echo "[INFO] Building the docker image and starting the container in the background..."
        pushd ${SCRIPT_DIR} > /dev/null 2>&1
        docker compose --file docker-compose.yaml up --detach --build --remove-orphans
        popd > /dev/null 2>&1
        ;;
    enter)
        echo "[INFO] Entering the existing 'ai_olympics' container in a bash session..."
        pushd ${SCRIPT_DIR} > /dev/null 2>&1
        docker exec --interactive --tty ai_olympics bash
        popd > /dev/null 2>&1
        ;;
    stop)
        echo "[INFO] Stopping the launched docker container..."
        pushd ${SCRIPT_DIR} > /dev/null 2>&1
        docker compose --file docker-compose.yaml down
        popd > /dev/null 2>&1
        ;;
    *)
        echo "[Error] Invalid argument provided: $1"
        print_help
        exit 1
        ;;
esac
