# Fed-BioMed release HOWTO

Make coordinated release of software (this repo) and documentation (published on `http://fedbiomed.org`)
  * using same version tag
  * corresponding to same software version
  * **release software before documentation** (for API documentation)
  * release at barely the same time

Release principle: follow the [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) release workflow. Can use `git` commands (see example below) or `git-flow` commands

## pre-release

- it is recommended to use a fresh clone of the repository to avoid any side effect

- checkout to `develop` branch, push your last changes, sync your local clone with remote
```bash
git checkout develop
git pull --prune
git commit
git push origin develop
```
- check that the CI for `develop` builds correctly on https://ci.inria.fr/fedbiomed/
- set the release version tag for the release (or use this tag directly in commands)
```bash
export RELEASE_TAG=v4.4
```
- fork a `release` branch from `develop`, and checkout the `release` branch
```bash
git checkout -b release/$RELEASE_TAG
```

## create release

- note: it is not needed to push to branch to the remote, as we currently don't have an additional step of multi-people test of the release branch
- in the `release` branch, do the release time updates:
  * `CHANGELOG.md`
  * `README.md` : change `v4.` occurences ; change `version 4` if major release
  * `fedbiomed/common/constants.py` : change `v4.`
- in the `release` branch, commit the release time updates
```bash
git commit -a
```

## merge release to master + create version tag

- checkout to `master` branch, sync your local clone with remote
```bash
git checkout master
git pull -p
```
- merge the `release` branch into `master`
```bash
git merge release/$RELEASE_TAG
```
- if merge conflicts occur, solve them
- create a version tag for the release
```bash
git tag -a $RELEASE_TAG
```
- push the updated `master` and tag to the remote
```bash
git push origin master
git push origin $RELEASE_TAG
```
- check that the CI builds correctly on https://ci.inria.fr/fedbiomed/
  * review carefully the log details for the build


## merge release to develop

- checkout to `develop` branch, sync your local clone with remote
```bash
git checkout develop
git pull -p
```
- merge the `release` branch into `develop`
```bash
git merge release/$RELEASE_TAG
```
- if merge conflicts occur, solve them
- push the updated `develop` to the remote
```bash
git push origin develop
```
- check that the CI builds correctly on https://ci.inria.fr/fedbiomed/


## cleanup

- delete the (local) release branch
```bash
git checkout develop
git branch -d release/$RELEASE_TAG
```
