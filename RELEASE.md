# Fed-BioMed release HOWTO

This procedure makes release of software and documentation (published on `https://fedbiomed.org`).
Both are contained in this repository.

Release principle: follow the [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) release workflow. Can use `git` commands (see example below) or `git-flow` commands

## pre-release

- it is recommended to use a fresh clone of the repository to avoid any side effect

- checkout to `develop` branch, sync your local clone with remote
```bash
git checkout develop
git pull --prune
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

## create release + create version tag

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

## merge release to master

- in github create a pull request for `release/$RELEASE_TAG` to `master`
  * one can auto-assign the PR, and doesn't need a review for this PR
- after checks successfully complete, please review the checks logs

- create a version tag for the release
  ```bash
  git tag -a $RELEASE_TAG
  ```
- push the tag to the remote
  ```bash
  git push origin $RELEASE_TAG
  ```

- do the merge
  *  pushing to master triggers the build action for documentation main pages such as `pages`, `support`, `news`.
  * check carefully the logs of the build pipeline in `Publish MASTER fedbiomed/fedbiomed.github.io` https://github.com/fedbiomed/fedbiomed/actions/workflows/doc-github-io-main-build.yml
- if merge conflicts occur, solve them

- check that the documentation pipeline completes successfully
  * new version of documentation is published after a new version tag is pushed. This action builds documentation related contents which are located in `docs/getting-started`, `docs/user-guide`, `docs/developer`, `docs/tutorials`.
  * `Publish NEW TAG in fedbiomed/fedbiomed.github.io` https://github.com/fedbiomed/fedbiomed/actions/workflows/doc-github-io-version-build.yml builds correctly
  * review carefully the log details for the build
- browse a few pages in the new documentation on `https://fedbiomed.org` to verify it works as expected

- optionally sync your local clone of `master` with new version of remote
```bash
git checkout master
git pull -p
```

## merge release to develop

- since the release branch was merged into `master`, it was deleted on remote. There are two options for restoring the branch:
  - either in github, go to the closed PR (the one opened for `master`) and click on "Restore branch."
  - or checkout the `release/$RELEASE_TAG` branch and push it again to re-create on the remote
    ```bash
    git checkout release/$RELEASE_TAG
    git push
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

# Fed-BioMed hotfix HOWTO

A hotfix is a patch that is added as a new version patch without changing the API or adding new features, when it can't wait for the next release to be corrected. To apply a hotfix, follow these steps:

This procedure makes hotfix of software and documentation (published on `https://fedbiomed.org`).
Both are contained in this repository.

Release principle: follow the [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) hotfix workflow. Can use `git` commands (see example below) or `git-flow` commands

## pre-hotfix

- it is recommended to use a fresh clone of the repository to avoid any side effect

- checkout **to `master` branch** (**Warning: hotfix are not created from `develop` !**), sync your local clone with remote

  ```bash
  git checkout master
  git pull --prune
  ```

- check that the CI for `master` builds correctly (github checks)
- choose a name (eg `521-short-description`) for the issue and assign it to `$HOTFIX_NAME`

  ```bash
  export HOTFIX_NAME=521-short-description
  ```

- set the hotfix version tag for the release (or use this tag directly in commands).For example, if the previous version was `v4.4.0`, it becomes `v4.4.1`.

  ```bash
  export HOTFIX_TAG=v4.4.1
  ```

- fork a `hotfix/$HOTFIX_NAME` branch from `master`, and checkout the `hotfix/$HOTFIX_NAME` branch

  ```bash
  git checkout -b hotfix/$HOTFIX_NAME
  ```

## create hotfix + create version tag

- apply the changes for the hotfix

- in the `hotfix/$HOTFIX_NAME` branch, do the hotfix time updates:
  * `CHANGELOG.md`
  * `fedbiomed/common/constants.py` : change `__version__`
- in the `hotfix` branch, commit the hotfix time updates

  ```bash
  git commit -a
  ```

- push the updated `hotfix/$HOTFIX_NAME`

  ```bash
  git push origin hotfix/$HOTFIX_NAME
  ```

## merge hotfix to master

- in github create a pull request for `hotfix/$HOTFIX_NAME` to `master`
  * one can auto-assign the PR, **but a reviewer should approve it before merging** (different from a release where the code in the PR was already reviewed !)
- after checks successfully complete, please review the checks logs

- create a version tag for the hotfix

  ```bash
  git tag -a $HOTFIX_TAG
  ```
- push the tag to the remote

  ```bash
  git push origin $HOTFIX_TAG
  ```

- do the merge
  *  pushing to master triggers the build action for documentation main pages such as `pages`, `support`, `news`.
  * check carefully the logs of the build pipeline in `Publish MASTER fedbiomed/fedbiomed.github.io` https://github.com/fedbiomed/fedbiomed/actions/workflows/doc-github-io-main-build.yml
- if merge conflicts occur, solve them

- check that the documentation pipeline completes successfully
  * new version of documentation is published after a new version tag is pushed. This action builds documentation related contents which are located in `docs/getting-started`, `docs/user-guide`, `docs/developer`, `docs/tutorials`.
  * `Publish NEW TAG in fedbiomed/fedbiomed.github.io` https://github.com/fedbiomed/fedbiomed/actions/workflows/doc-github-io-version-build.yml builds correctly
  * review carefully the log details for the build

- browse a few pages in the new documentation on `https://fedbiomed.org` to verify it works as expected

- optionally sync your local clone of `master` with new version of remote

  ```bash
  git checkout master
  git pull -p
  ```

## merge hotfix to develop

- since the hotfix branch was merged into `master`, it was deleted on remote. There are two options for restoring the branch:
  - either in github, go to the closed PR (the one opened for `master`) and click on "Restore branch."
  - or checkout the `hotfix/$HOTFIX_NAME` branch and push it again to re-create on the remote

    ```bash
    git checkout hotfix/$HOTFIX_NAME
    git push
    ```

- in github create a pull request for `hotfix/$HOTFIX_NAME` to `develop`
  * one can auto-assign the PR, and doesn't need a review for this PR
- after checks complete, please review the checks logs
- do the merge
- if merge conflicts occur, solve them

## cleanup

- delete the (local) release branch
  ```bash
  git checkout develop
  git branch -d hotfix/$HOTFIX_NAME
  ```

# Publishing news, fixing documentation

The changes in website that do not affect `User documentation` part are also considered as hotfix. These modifications involve:

* Adding news or updating their content
* Adding new items to front page (button, boxes, description, new section)
* Adding new pages (These are pages NOT related to user documentation such as `About Us`, `Roadmap` etc.) 
* Adding new items for footer area
* Changing contents of static pages About us, Contributors etc.
* Front-End issues broken layout etc.

However, since these modifications don't contain any changes in the Fed-BioMed source code, it is not considered as code patch. It means after applying the changes new version tag **SHOULD NOT BE** pushed. The process flow is the same as `hotfix` except that one doesn't push a new tag. 

Also, for publishing a news or a doc fix, a review by a third party is not mandatory.

- Create a new branch `hotfix/$HOTFIX_NAME` **from the `master` branch**.
- Apply your web-site related changes (please make sure that you change files only in `docs` directory).
- Create a pull request (PR) to merge the hotfix branch **into the `master` branch** and wait for all the checks to pass.
- Merge `hotfix` branch also into `develop` branch. You can follow the instructions that are explained in hotfix section


