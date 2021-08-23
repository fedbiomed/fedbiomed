# Fedbiomed network

## Introduction

This repo provides :

* a RESTful API for uploading models and data associated to fedbiomed
* a MQTT service

These two services are provided as docker containers

## Installation

```bash
source ./scripts/fedbiomed_environment network
cd ./envs/development/network
```

* on localhost
```bash
./deploy.sh --local
```

* on a networked or vpn'ized install (not verified yet)
```bash
./deploy.sh
```
