# Developer info on continuous integration

Continuous integration uses [GitHub Actions](https://github.com/fedbiomed/fedbiomed/actions). 

## Events that trigger CI tests

CI tests are triggered automatically by GitHub on a:

- pull request to `develop` or `master` branch
- push in `develop`, `master`, `feature/test_ci` branches (eg: after a merge, pushing a fix directly to this branch)


The pull request can not be completed before CI pipeline succeeds

- pushing a fix to the branch with the open pull request re-triggers the CI test
- CI test can also be manually triggered form `Pull Requests` > `Check` > `Re-run all checks` or directly from `Action` tab. 

CI pipeline currently contains :

- running unit tests
    - update conda envs for `network` and `researcher`
    - launch nework
    - run unit tests

- running a simplenet + federated average training, on a few batches of a MNIST dataset, with 1 node. For that, CI launches `./scripts/run_test_mnist` (an also be launched on localhost)
    - update conda env for `node` (rely on unit tests for others)
    - activate conda and environments, launch network and node.
    - choose an existing git branch for running the test for each of the repos, by decreasing preference order : source branch of the PR, target branch of the PR, `develop`
    - launch the `fedbiomed` script `./notebooks/101_getting-started.py`
    - succeed if the script completes without failure.

- running test build process for documentation 


!!! "note" Execution exceptions 
    CI build tests are run if a file related to the build is changed. For example, if the changes (difference between base and feature branch) in a pull request are only made in the gui directory or docs, the CI action for unit tests will be skipped. Please see the exceptions in `.gihub/workflows/*.yml`

## Displaying Outputs and Results

To view CI test output and logs:

- view the pull request in github (select `Pull requests` in top bar, then select your pull request).
- click on the `Checks` at the top bar of the pull request and select the `Check` that you want to display.
- Click on the jobs to see its console output. 

### Unit tests coverage 

Unit tests coverage reports are published on Codecov platform for each branch/pull request. The report contains overall test coverage for the branch and detailed coverage rates file by file. 

- Once a GitHub workflow/pipeline is executed for unit-test Codecov with automatically add a comment to the pull request that shows:
    - Overall test coverage
    - The difference code coverage between base and feature branch 

To access reports on Codecov please go [Fed-BioMed Codecov dashboard](https://app.codecov.io/gh/fedbiomed/fedbiomed/) or go to your pull request,click on `Checks` at the top of the pull request view and click on `View this Pull Request on Codecov`


## CI and GitHub Actions Configuration


GitHub actions are configured using `yml` files for each workflow. Workflow files can contain multiple jobs and multiple steps for each job. Please go `.github/workflow` directory to display all workflows for CI. 

The `name` value in each `yml` file corresponds to the name of the workflows that are displayed in `Actions` [page of the Fed-BioMed repository](https://github.com/fedbiomed/fedbiomed/actions). The `name` value under each `job` corresponds to each `Checks` in pull requests.

Please see [GitHub actions](https://github.com/features/actions) documentation for more information. 

### CI slaves

CI slaves are located on `ci.inria.fr`. To be able to add extra configuration and installation you have to connect with your account on `ci.inria.fr`. You need to be approved by one member of the Fed-BioMed CI project or to be a member of Inria to be able get an account on `ci.inria.fr`. You can request the Fed-BioMed team to become a member of the Fed-BioMed CI project.


## Testing

Using branch `feature/test_ci` can be useful when testing/debugging the CI setup (triggers CI on every push, not only on pull request).


More integration tests run on a nightly basis. They need a conda environment `fedbiomed-ci.yaml` which can be found in `./envs/ci/conda`
