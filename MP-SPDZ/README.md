# MPSPDZ for Multi Party Computation 

This directory contains scripts and binaries for multi party computation using MP-SPDZ library in Fed-BioMed. 


## MPC Protocols 

Currently, the only MPC protocol used in Fed-BioMed is shamir protocol to calculate server key for secure aggregation. 
The binary files of the protocols are located in the directories called `Linux_amd64` and `Linux_avx2` that contains 
same protocols for different CPU architectures. These binaries can only be used in Linux based operating systems. Please 
see the section "Rebuilding Binaries" to re-build binary files from MP-SPDZ source distribution. 


### Notes for macOS Installation

For macOS, the installation should be done through source distribution of MP-SPDZ. This directory does not contain binaries
for macOS (Darwin). The script located in `<FEDBIOMED_DIR>/scripts/fedbiomed_configure_secagg` are created to configure 
MP-SPDZ, as well as installing MP-SPDZ from source distribution if the operating system is Darwin. Runing this
script on macOS will be enough for installing and configuring MP-SPDZ. 

## MPC Scripts

This directory also contains multi-party computation script that can be executed on `shamir` or other protocols. This
MPC scripts can only be compiled using MP-SPDZ. Therefore, please make use that MP-SPDZ is installed and configured 
before the compilation. Please see `<FEDBIOMED_DIR>/scripts/fedbiomed_mpc (node|researcher) *WORKDIR* compile --help` for compilation instructions.


## Re-Building MPC Protocols 

It is highly recommended to use Docker container to re-build binaries for Linux based operating system in order to 
avoid issue due to dependencies and mismatch versions. Please follow the steps below to re-build binaries. 

### 1. Fresh clone of MP-SPDZ Repository

Before starting re-building binaries, please use fresh clone of MP-SPDZ instead of using the MP-SPDZ submodule of 
Fed-BioMed, and make sure that the MP-SPDZ in your fresh clone is not compiled locally. 

**Note:** This document is prepared and tested with MP-SPDZ version v0.3.4.

```shell
cd <FEDBIOMED_DIR>
git clone https://github.com/data61/MP-SPDZ.git MP-SPDZ_build --branch v0.3.4
cd MP-SPDZ_build
```

### 2. Build MP-SPDZ Image 

Create base docker image for MP-SPDZ using the command below. This command will build the base image where all necessary 
dependencies are installed for MP-SPDZ. This process that approximately 10min.

```shell
docker build --tag mpspdz:buildenv --target buildenv .
```

### 3. Run MP-SPDZ Container and Copy Build Script

After the image is ready, you run following command to start MP-SPDZ container called `fedbiomed-mpspdz-dev`. 

```shell
docker run -d --name fedbiomed-mpspdz-dev mpspdz:buildenv tail -f /dev/null
```

After the container is started, you can copy the build script from your host to docker container. This script is 
configured for building required MP-SPDZ binaries for Fed-BioMed. 

```shell
# Assumes that you are in the directory `MP-SPDZ_build`
docker cp ../MP-SPDZ/build_shamir_2048bits fedbiomed-mpspdz-dev:/usr/src/MP-SPDZ/
```
### 4. Execute Build Script  

Please run following command to build binaries. This process takes between 5-10min.  

```shell
docker exec -it fedbiomed-mpspdz-dev bash ./build_shamir_2048bits
```

### 5. Copy Binaries

As the final step, after the build operation is finished, the binaries that are created in the docker container 
should be copied into `MP-SDPZ` directory of Fed-BioMed. The `bin` directory contains binaries for different 
CPU architecture. Therefore, please pay attention to copy the binaries into correct CPU architecture folders in `MP-SDPZ`
directory of Fed-BioMed. 

```shell
# Assumes that you are in the directory `MP-SPDZ_build`

# Linux AMD64
docker cp fedbiomed-mpspdz-dev:/usr/src/MP-SPDZ/bin/Linux-amd64/shamir-party.x ../MP-SPDZ/Linux-amd64/shamir-party.x

# LINUX AVX2
docker cp fedbiomed-mpspdz-dev:/usr/src/MP-SPDZ/bin/Linux-avx2/shamir-party.x ../MP-SPDZ/Linux-avx2/shamir-party.x
```

### Testing the Binaries

The script `fedbiomed_configure_secagg` should be run in order to reconfigure MP-SPDZ with new binaries and test them. 

```shell
scripts/fedbiomed_configure_secagg node 
```
or
```shell
scripts/fedbiomed_configure_secagg researcher 
```

If the binaries created correctly and there is no error of configuration, you will be seeing the message 
that says "MP-SPDZ configuration is successful!"