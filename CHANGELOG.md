# Fed-BioMed changelog

## 2022-xx-xx version 3.6 or 4.0

- TrainingArgs class to manage/verify training arguments on researcher side

## 2022-05-09 version 3.5

- add FedProx optimization scheme support for PyTorch in training plan
- data manager to provide robust solution on managing node datasets before training on the node side
- model evaluation/testing on the node side over locally updated and aggregated model parameters
- add NIFTI folder dataset type
- add option to load MedNIST dataset in node
- update docstrings for API documentation support
- node container support for GPU with PyTorch, tested on Fedora 35
- debug and robustify VPN/containers environment, test on Ubuntu 20 and Windows 10 + WSL2
- VPN/containers environment scripts for simpler management of containers and peers
- refactor training\_plans/\_fedbiosklearn.py to get rid off eval()
- change training\_plans file organisation
- create a top class for training\_plans
- removed the magic %writefile use in notebooks to save the user's defined model
- more unittests and flake8 parsing
- add validation class for checking user input

## 2022-02-25 version 3.4

- rewriting of Opacus notebook
- new tutorial notebook on Experiment() usage
- add .coveragerc to tune test coverage
- fix mqtt logger loop then mqqt not reachable
- replace @property getters/setters by proper getters() setters(), still in progress
- a lot of new unit tests and increase test coverage
- refactor of Message() class, simplication of Messages description (purely declarative now)
- add more ErrorNumbers + associated messages
- check user input (mainly in Experiment() for now)
- rename Exceptions as Errors, add FedbiomedError as top class of our errors
- use try/except block at low level layers
- Environ() class refactoring, environment tests rewriting
- Experiment() class refactoring, new API, more setters/getters, interactive use, rename rounds -> round_limit,...
- add single GPU training support for PyTorch
- add a gui to manage data on nodes
- update of sklearn sgdregressor notebook
- update of monai notebook
- Tensorboard fixes for multi class classification with scikit learn

## 2022-01-07 version 3.3

- add MONAI support and example notebooks
- add model manager to register and check authorized training models on the node based on model hash
- refactor experiment real time monitoring capacity with Tensorboard
- add `Request.list()` to list shared dataset on online nodes
- configure_conda may take parameters to only update some environments
- fix conda environments for mac OSX
- add -n (dryrun) option for configure_conda, for debug/validation purpose
- fix and refactor breakpoint feature which was not fully operational
- change the names of breakpoint directories
- node error reporting to researcher
- basic error handling on researcher component
- mutualize the Singleton metaclass
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
