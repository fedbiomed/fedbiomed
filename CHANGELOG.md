# Fed-BioMed changelog

## 2025-05-16 version 6.1.0

- Update testing/validation functionality to use fixed subset of data for testing on the node side between training rounds.
- Add support for Docker image versioning; versions can now be specified during image build to allow multiple Fed-BioMed image versions.
- Starts Fed-BioMed component automatically in docker when docker containers are launched.
- Improve Docker networking by assigning a separate network to each Fed-BioMed component.
- Updates containers to use `data` directory within component main folder by default.
- Fix tab activation issue in the Node GUI on the dataset add page.
- Fix broken documentation links caused by overwritten MkDocs redirection rules.
- Updates windows installation documentation.

## 2025-02-06 version 6.0.0

- In addition to the release candidate (details below), fix issue #1248, spaces where wrongly added when adding multiple tags in the GUI
- Fix issue #1105 about incorrect parsing of `niftii` image file's name in the medical folder dataset.

## 2025-02-05 version 6.0.0.rc1

- Implementation of packaging for the library/framework to publish on the PyPI repository.
- CI/CD: configuration for continues deployment.
- Creation of the `fedbiomed` executable, installed alongside the Fed-BioMed installation.
- Documentation updates following related changes.
- Introduction of the PDM package manager for development.
- Updates to Docker containers to eliminate the use of `miniconda`.
- Introduction of VPN Docker container tests.
- Removal of the `environ` class in favor of the improved `Config` class.
- Secure aggregation implementation for advanced federated optimization functionalities.
- Fixes: 'colab' shell recognition issue #1274

## 2025-01-24 version 5.4.1

- Patch for missing list request handler on the node side.

## 2024-11-04 version 5.4.0

- introduce in-application implementation of Shamir Additive Secret Sharing (ASS) for Joye-Libert secure aggregation
- remove dependency to MP-SPDZ library, using in-app ASS
- introduce node to node messages security for honest but curious scenario to support in-app ASS
- improve node to node messages implementation to use symmetric encryption
- refactor `Message` class, improving API to upper layers
- fix researcher jupyter notebook automatic token transmission not working
- fix LOM secagg researcher side aggregation error for values >= 2**31
- misc give clearer node side error message in case of overflow with LOM secagg

## 2024-09-05 version 5.3.0

- introduce fast secure aggregation using Low-Overhead Masking (LOM) scheme and in-app Diffie-Hellman key exchange
- add feature for choosing the batch size for testing (validation) facility
- add transfer learning tutorial
- fix make optional potentially insecure consistency check (validation) for secure aggregation
- fix issue with validation when `test_batch_size=1`
- multiple fixes in documentation and tutorials

## 2024-05-27 version 5.2.0

- redesign `Job` and `Experiment` classes, introducing `federated_workflow` class
- introduce end-to-end and testing facility
- fix, robustify and redesign gRPC communication layer
- add a concrete-ml example
- improve logging facility
- fix SecAgg (`Experiment` with more 4 nodes or more was failing)
- fix quantization process for secure aggregation
- review notebooks. Remove unrelevant ones.
- fix and extend `model` class, by handling buffers layers such as batch normalization layers.
- multiple fixes for breakpoint
- add a [concrete-ml](https://docs.zama.ai/concrete-ml) example

## 2024-02-19 version 5.1.0

- multiple fixes and improvements for gRPC communications stability
- fix scaffold failure when reloading breakpoint and continue training
- fix breakpoint save failure for DecLearn optimizers
- fix import/export model check for explicit error when not instantiated

## 2024-01-26 version 5.0.2

- fix crash happening when the tkinter file selector is used in non-interactive mode

## 2024-01-18 version 5.0.1

- fix `fedbiomed_vpn` failure due to time race under certain circumstance

## 2023-12-21 version 5.0.0

- introduce new communication architecture based on gRPC
- introduce communication TLS encryption for automatic protection from passive MITM
- redesign researcher side's `Requests` for better performance and robustness
- add support for weighted average with secure aggregation
- fix secure aggregation bug when using more than 2 nodes
- fix tune DecLearn sample notebook parameters for demonstrating high accuracy
- fix misc error in checking coherence of node side validation parameters
- fix misc errors in notebooks

## 2023-11-09 version 4.5.0

- improve import interface for DecLearn optimizers
- improve training plan dependencies import on researcher side
- fix error when using global model node side testing due to node state bug
- fix node GUI dependencies
- fix `configure_conda` failing silently
- fix misc issues in documentation and notebooks
- fix web site download link for tutorial notebooks

## 2023-09-18 version 4.4.4

- fix failure due to erroneous researcher side data quality checks: remove currently useless data quality checks

## 2023-08-02 version 4.4.3

- fix support for per-layer learning rate for PyTorch models

## 2023-07-27 version 4.4.2

- update software dependencies and implement more strict package version checks
- fix docker containers build failures due to outdated package dependencies

## 2023-06-23 version 4.4.1

- fix secure aggregation vector encoding bug

## 2023-06-05 version 4.4.0

- add HTTPS secure access to Fed-BioMed Node GUI
- introduce GitHub workflow/actions for CI build tests and testing/publishing documentation.
- introduce versioning for component config files, MQTT messages and breakpoints
- migrate to GitHub
- migrate `docs` source into main repository and point to https://fedbiomed.org
- fix robustness of handling secure aggregation training errors.
- fix warnings in `TorchModel.set_weights` and BatchNorm layers' handling
- fix incorrect calculation of SkLearn model weights
- fix incorrect compatibility of FedProx feature with `Model` class
- fix ordering of weights and node replies after training.

## 2023-04-26 version 4.3

- introduce secure aggregation using Joye-Libert scheme and Shamir MPC key computation
- update MONAI and scikit-learn version used
- fix Scaffold incorrectly applying correction states
- fix incorrect Perceptron default values for scikit-learn models
- fix `Experiment.set_training_args()` not propagating updated value
- fix environment cleaning to handle configuration file content change
- fix docker wrapping scripts to restrict container account names to alphanumeric characters
- misc improve node CLI for non-interactive add of MedicalFolderDataset using a json file

## 2023-02-08 version 4.2

- add support for `docker compose` v.2 file syntax
- fix model weights computation occurring during aggregation, by sending dataset sample size from node to researcher
- fix GUI regression failure, after merging MP-SPDZ certificate generation - such issue was freezing some web browsers
- fix incoherent tag handling: make explicit the way datasets are tagged on nodes
- fix unit tests failure, when launched from root directory, due to missing mocking facility
- fix `fedbiomed_run` error: prevent launching researcher when no config file exists
- misc improve make sure only one dataset per Node is selected during the training
- misc remove uncorrect warning about `optimizer_args` when using SKlearn training plan

## 2023-01-05 version 4.1

- introduce Scaffold implementation for PyTorch
- introduce training based on iteration number (`num_updates`) as an alternative to epochs
- introduce provisions for including external contributors in the project
- add nightly continuous integration test of selected notebooks
- add documentations for network matrix and security model
- update image segmentation notebook to match documentation tutorial
- add round number in researcher side training progress message
- fix MedicalFolderDataset with demographics file using column 0 as subject folder key
- fix batch size display when using Opacus
- fix loss display when using FedProx
- fix default value of `batch_maxnum` to a reasonable value of 0
- fix validation with custom metrics backend code and validation example notebook
- fix image segmentation notebook typo
- misc improve MedicalFolderDataset with a reasonable default value for `demographics_transform`
- misc improve error message when dataset geometry does not meet researcher side quality check

## 2022-11-17 version 4.0

- introduce IXI (image + CSV file) dataset support as MedicalFolderDataset
- add advanced brain image segmentation tutorial for IXI dataset
- add node side GUI support for IXI dataset
- major redesign of training plan implementation for genericity
- redesign Opacus integration with torch training plan
- implement central and local differential privacy (CDP/LDP) for Pytorch training plan
- introduce integration with FLamby FL benchmark package
- introduce node side GUI user accounts, authentication, accounts management
- introduce data loading plan functionality for dataset load-time custom view on the node side
- add data loading plan support for IXI medical folder dataset
- introduce training plan approval capability in application: researcher request, node approval CLI
- add node side GUI support for training plan approval
- introduce mini-batch support in scikit-learn training plans
- refactor scikit-learn training plans with hierarchical design
- refactor NIFTI folder dataset type for code quality and robustness
- TrainingArgs class to manage/verify training arguments on researcher side
- rename model approval as training plan approval for coherency
- add sample notebook for researcher-side filtering of datasets on minimum samples number
- obfuscate node side path to researcher for better privacy
- misc node side TinyDB database access refactor for code quality and robustness
- misc improve scikit-learn training plan dependency handling
- fix bug on training plan report of sample/percentage progress
- fix missing `fedprox_mu` parameter in training args
- fix dry run mode for pytorch training plan
- fix conda environment GLIBC version issue

## 2022-05-09 version 3.5

- add FedProx optimization scheme support for PyTorch in training plan
- data manager to provide robust solution on managing node datasets before training on the node side
- model evaluation/validation on the node side over locally updated and aggregated model parameters
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
