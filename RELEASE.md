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
  * `fedbiomed/common/constants.py` : change `__version__`
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

## Hotfix 

A hotfix is a patch that is added as a new version patch without changing the API or adding new features. To apply a hotfix, follow these steps:

- Create a new branch `hotfix/...` from the `master` branch.
- Apply your hotfix and commit the changes.
- Update the `CHANGELOG.md` file with the new patch version. For example, if the previous version was `v4.0.0`, it becomes `v4.0.1`.
- Create a pull request (PR) to merge the hotfix branch into the `master` branch and wait for all the checks to pass.
- Ask for someone to review and approve
- Push the new tag and save it:
  - Set the tag name: `export HOTFIX_TAG=v4.0.x`
  - Add the tag: `git tag add $HOTFIX_TAG`
  - Push the tag to the remote repository: `git push origin $HOTFIX_TAG`

- After the new tag is pushed, go to the PR and merge it into the `master` branch.
- The hotfix should also be merged into the `develop` branch:
  - Since the hotfix branch was merged into `master`, it may have been deleted. It needs to be restored in order to merge it into the `develop` branch.
  - There are two options for restoring the branch:
     - Go to the closed PR (the one opened for `master`) and click on "Restore branch."
     - If you have the local `hotfix` branch, you can push it to the remote repository again.
  - Create a new PR to merge the `hotfix` branch into the `develop` branch.
  - Merge it after all the checks have been run.

- Go `fedbiomed.org` documentation site to verify that new version has been added. Please note that the documentation deployment has been configured to display only the latest patch of a version.

## Publishing news, adding new pages or updating existing pages

The changes in website such as adding new pages, news or updating existing ones are considered as hotfix. However, since these modifications don't contain any changes in the Fed-BioMed source code, it is not considered as new patch. It means after applying the changes new version tag **SHOULD NOT BE** pushed. The process flow is the same as `hotfix` except pushing a new tag. 

- Create a new branch `hotfix/...` from the `master` branch.
- Apply your web-site related changes (please make sure that you change files only in `docs` directory).
- Create a pull request (PR) to merge the hotfix branch into the `master` branch and wait for all the checks to pass.
- Merge `hotfix` branch also into `develop` branch. You can follow the instructions that are explained in [hotfix](#hotfix) section