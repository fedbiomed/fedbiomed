# Developer info on continuous integration

Continuous integration uses a [Jenkins](https://www.jenkins.io/) server on `ci.inria.fr`. 

CI tests are triggered automatically by gitlab on a :

* merge request to `develop` or `master` branch
* push in `develop`, `master`, `feature/test_ci` branches (eg: after a merge, pushing a fix directly to this branch)

The merge should not be completed before CI pipeline succeeds

* pushing a fix to the branch with the open merge request re-triggers the CI test
* CI test can also be manually triggered by adding a comment to the merge request with the text `Jenkins please retry a build`

CI pipeline currently contains :

* running unit tests
* running a simplenet + federated average training, on a few batches of a MNIST dataset, with 1 node. For that, CI launches `./scripts/CI_build` (wrapping for running on CI server) which itself calls `./scripts/run_test_mnist` (payload, can also be launched on localhost)

  - clone the Fed-BioMed repository, set up condas and environments, launch network and node. 
  - choose an existing git branch for running the test for each of the repos, by decreasing preference order : source branch of the merge, target branch of the merge, `develop`
  - launch the `fedbiomed` script `./notebooks/getting-started.py`
  - test succeeds if the script completes without failure.


To view CI test output and logs :

* view the merge request in gitlab (select `Merge requests` in left bar, then select your merge request)
* click on the `Pipeline` number (eg: #1289345) in the merge request, then click on the `Jobs` tab, then click on the job number (eg: #1294521)
* select `Console output` in the left pane

To configure CI test :

* connect with your account on `ci.inria.fr`. To get an account on `ci.inria.fr` you need to be approved by one member of the Fed-BioMed CI project or to be a member of Inria
* request the Fed-BioMed team to become a member of the Fed-BioMed CI project

Note: using branch `feature/test_ci` can be useful when testing/debugging the CI setup (triggers CI on every push, not only on merge request).

More integration tests run on a nightly basis. They need a conda environment `fedbiomed-ci.yaml` which can be found in `./envs/ci/conda`