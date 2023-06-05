# Fed-BioMed release HOWTO

This procedure make release of software and documentation (published on `https://fedbiomed.org`).
Both are contained in this repository.

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
- check that the CI for `develop` builds correctly (github checks)
- set the release version tag for the release (or use this tag directly in commands)
```bash
export RELEASE_TAG=v4.4.0
```
- fork a `release/$RELEASE_TAG` branch from `develop`, and checkout the `release/$RELEASE_TAG` branch
```bash
git checkout -b release/$RELEASE_TAG
```

## create release

- note: it is not needed to push to branch to the remote, as we currently don't have an additional step of multi-people test of the release branch
- in the `release/$RELEASE_TAG` branch, do the release time updates:
  * `CHANGELOG.md`
  * `fedbiomed/common/constants.py` : change `__versions__`
- **add new version news in documentation**
- in the `release` branch, commit the release time updates
```bash
git commit -a
```
- push the updated `release/$RELEASE_TAG`
```bash
git push origin release/$RELEASE_TAG
```

## merge release to master + create version tag

- in github create a pull request for `release/$RELEASE_TAG` to `master`
  * one can auto-assign the PR, and doesn't need a review for this PR
- after checks complete, please review the checks logs
- do the merge
  *  pushing to master triggers the build action for documentation main pages such as `pages`, `support`, `news`.
  * check carefully the logs of the build pipeline in `Publish MASTER fedbiomed/fedbiomed.github.io` https://github.com/fedbiomed/fedbiomed/actions/workflows/doc-github-io-main-build.yml
- if merge conflicts occur, solve them

- checkout to `master` branch, sync your local clone with remote
```bash
git checkout master
git pull -p
```
- create a version tag for the release
```bash
git tag -a $RELEASE_TAG
```
- push the tag to the remote
```bash
git push origin $RELEASE_TAG
```
- check that the documentation pipeline completes successfully
  * new version of documentation is published after a new version tag is pushed. This action builds documentation related contents which are located in `docs/getting-started`, `docs/user-guide`, `docs/developer`, `docs/tutorials`.
  * `Publish NEW TAG in fedbiomed/fedbiomed.github.io` https://github.com/fedbiomed/fedbiomed/actions/workflows/doc-github-io-version-build.yml builds correctly
  * review carefully the log details for the build

- browse a few pages in the new documentation on `https://fedbiomed.org` to verify it works as expected


## merge release to develop

- checkout the `release/$RELEASE_TAG` branch and push it again to re-create on the remote
```bash
git checkout release/$RELEASE_TAG
git push origin release/$RELEASE_TAG
```

- in github create a pull request for `release/$RELEASE_TAG` to `develop`
  * one can auto-assign the PR, and doesn't need a review for this PR
- after checks complete, please review the checks logs
- do the merge
- if merge conflicts occur, solve them

## cleanup

- delete the (local) release branch
```bash
git checkout develop
git branch -d release/$RELEASE_TAG
```

