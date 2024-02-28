# Troubleshooting Fed-BioMed 

## Mac M1-2-3 Issues

### fedbiomed/vpn-base or fedbiomed/vpn-basenode image build errors due to arm64 system

By default, the docker may use the images that are created for arm64/aarch64. The docker build files of the fedbiomed images and the libraries that are installed within are compatible for amd64 type of platforms. Therefore, you may get some error while `fedbiomed_vpn` script builds some images. Those error can be during the miniconda installation, secure aggregation setup or while installing some of required pip dependencies for Fed-BioMed environment.  You can fix this problem by setting envrionment variable that declares default default docker platform to force docker to use linux/amd64. 

```
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

After setting this variable you can execute build command through ``{FEDBIOMED_DIR}/scripts/fedbiomed_run build`
