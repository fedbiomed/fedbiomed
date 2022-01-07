# Fed-BioMed changelog

## 2022-01-07 version 3.3

- add MONAI support and example notebooks
- add model manager to register and check authorized training models on the node
- refactor experiment real time monitoring capacity with Tensorboard
- add `Request.list()` to list shared dataset on online nodes
- configure_conda may take parameters to only update some environments
- fix conda environments for mac OSX
- add -n (dryrun) option for configure_conda, for debug/validation purpose
- fix and refactor breakpoint feature which was not fully operational
- change the names of breakpoint directories
- refactor environ as singleton class
- fix the way the tests deal with fedbiomed.common.environ
- refactor strategy (moved some methods in upper classes)
- add command **run_integration_test** to easily run an integration test from a single .py or .ipynb
- add an automatized method to add a dataset in nodes's db from a JSON dataset description file
- add error numbering as an enum, impact on error messages serialization
- more example notebooks, update existing notebooks
- more unittests
- normalize naming : use term 'node' not 'client'

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
