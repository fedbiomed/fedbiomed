# Fed-BioMed changelog

## 2021-10-21 version 3.2

- add support for scikit-learn with SGD regressor and example notebook
- add VPN + docker environment for deploying over an untrusted network
- add message logging capability including sending node messages to researcher
- add loss report from node during training and view in tensorboard in researcher
- add save/load state capability after each round during a training
- add capability for listing datasets on each node
- add example notebooks for Celeba and used cars dataset
- WIP add unit tests
- add support for multiple Experiment(), including re-executing a notebook
- fix issue erratic failure when training with 3+ nodes
- test and document Windows 10 installation in WSL

## 2021-08-13 version 3.1

- merge 3 gitlab repos fedbiomed-{network,node,researcher} in a unique fedbiomed repo
- add new model variational autoencoder (VAE)
- add support for generic dataloader and handling of .csv dataset
- measure execution time on nodes
- WIP adding unit tests
- misc code cleaning

## 2021-07-05 version 3.0

- initial release of re-implementation based on pytorch model file transfer and MQTT messaging

