# Fedbiomed network

## Introduction

This repo provides :

* a RESTful API for uploading models and data associated to fedbiomed
* a MQTT service

These two services are provided as docker containers

## Installation

### Clone the repo ussing:
```bash
git clone git@gitlab.inria.fr:fedbiomed/fedbiomed-network.git
```

### initialize the environment

```
# ../../../scripts/fedbiomed_environment network
```

* on localhost
```bash
cd fedbiomed-network
./deploy.sh --local
```


* on epione-demo (not verified yet)
```bash
cd fedbiomed-network
./deploy.sh
```
