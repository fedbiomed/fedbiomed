# Fed-BioMed release HOWTO

This procedure makes release of software and documentation (published on `https://docs.fedbiomed.org`).
Both are contained in this repository.

Release principle: follow the [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) release workflow. Can use `git` commands (see example below) or `git-flow` commands

**Key rule:** after anything is merged into `master` (release or hotfix), `master` itself is merged
back into `develop` (never the release or hotfix branch). This keeps `master` an ancestor of
`develop`, so a release branch created from `develop` is never behind `master`.

## Version files

Every release and every hotfix updates **all** of these files to the new version
(`X.Y.Z` below, without the `v` prefix unless shown). The packages pin each other
to the exact version, so a single outdated file makes the build fail
(`ResolutionImpossible` when installing the packages, e.g. in the Docker base image).

| File | Field |
| --- | --- |
| `packages/fedbiomed/src/fedbiomed/__init__.py` | `__version__ = "vX.Y.Z"` |
| `packages/fedbiomed/pyproject.toml` | optional dependencies `fedbiomed-node-api==X.Y.Z` and `fedbiomed-gui==X.Y.Z` |
| `packages/node-api/pyproject.toml` | `version = "X.Y.Z"` and dependency `fedbiomed==X.Y.Z` |
| `packages/node-api/src/fedbiomed_node_api/__init__.py` | `__version__ = "vX.Y.Z"` |
| `packages/node-ui/pyproject.toml` | `version = "X.Y.Z"` and dependency `fedbiomed-node-api==X.Y.Z` |
| `packages/node-ui/src/fedbiomed_gui/__init__.py` | `__version__ = "vX.Y.Z"` |
| `CITATION.cff` | `version: vX.Y.Z` |

If a package is added to `packages/`, add its version and pins to this table.
`packages/node-ui/ui/package.json` has its own frontend version and is not part of it.

On a branch created from an older `master` that predates `packages/`, only
`fedbiomed/__init__.py` and `CITATION.cff` exist: update those.

Check that no file was missed, with `OLD` the previous version:
```bash
export OLD=6.5.0
grep -rnF --include=pyproject.toml --include=__init__.py --include=CITATION.cff \
  "$OLD" packages CITATION.cff && echo "ERROR: $OLD still present" || echo "OK"
```

## pre-release

- it is recommended to use a fresh clone of the repository to avoid any side effect

- checkout to `develop` branch, sync your local clone with remote
```bash
git checkout develop
git pull --prune
```
- check that the scheduled CI for `develop` is green: `Fed-BioMed Full Build Test`
  (weeknights at 18:00 UTC) and `End-to-End Test` (weeknights at 23:00 UTC)
- check that `master` is fully merged into `develop` (the last release or hotfix
  must have been merged back with the "merge master to develop" procedure)
```bash
git fetch origin
git merge-base --is-ancestor origin/master origin/develop \
  && echo "OK" \
  || echo "ERROR: master is not merged into develop, do 'merge master to develop' first"
```
- set the release version tag for the release (or use this tag directly in commands)
```bash
export RELEASE_TAG=v6.1.0
```
- fork a `release/$RELEASE_TAG` branch from `develop`, and checkout the `release/$RELEASE_TAG` branch
```bash
git checkout -b release/$RELEASE_TAG
```

## create release + create version tag

- in the `release/$RELEASE_TAG` branch, do the release time updates:
  * `CHANGELOG.md`
  * every file of [Version files](#version-files): set the new version, then run
    the check command of that section

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

- create a version tag for the release (on the `release/$RELEASE_TAG` branch)
  ```bash
  git tag -a $RELEASE_TAG
  ```
- push the tag to the remote
  ```bash
  git push origin $RELEASE_TAG
  ```

- check that the documentation pipeline completes successfully
  * new version of documentation is published after a new version tag is pushed. This action builds documentation related contents which are located in `docs/getting-started`, `docs/user-guide`, `docs/developer`, `docs/tutorials`.
  * check that `Build and deploy docs to fedbiomed-docs-public` https://github.com/fedbiomed/fedbiomed/actions/workflows/build-and-deploy-documentation.yml builds correctly
  * review carefully the log details for the build

- check that the docker image publication completes successfully
  * pushing the version tag publishes the `fedbiomed/base`, `fedbiomed/node` and `fedbiomed/researcher` images to Docker Hub
  * check that `Tests and Publish Fed-BioMed Docker Images` https://github.com/fedbiomed/fedbiomed/actions/workflows/docker-deploy.yml completes correctly
  * verify the new version tag is listed on Docker Hub for the three images

- do the merge
  * the merge method MUST be "Create a merge commit" (never "Squash and merge"
    or "Rebase and merge")
  * the tag already published the package, the images and the documentation, so
    the merge publishes nothing. It triggers `End-to-End Test` on `master`,
    which verifies the released tree after the fact.
- if merge conflicts occur, solve them (should not happen when `master` was
  merged into `develop` before creating the release branch)

- browse a few pages in the new documentation on `https://docs.fedbiomed.org` to verify it works as expected

- optionally sync your local clone of `master` with new version of remote
```bash
git checkout master
git pull -p
```

## merge master to develop

`master` (not the release branch) is merged back into `develop`, so that the
merge commit created on `master` is also part of `develop`'s history. This
keeps `master` an ancestor of `develop`, so the next release branch is never
behind `master`.

A direct PR from `master` to `develop` can't be merged because of the
"Require branches to be up to date" rule. Use a temporary branch instead.
**Never click "Update branch" on a PR whose source is `master`**: it would merge
`develop` into `master`.

- create a temporary branch from `develop` and merge `master` into it
```bash
git fetch origin
git checkout -b merge/master-to-develop-$RELEASE_TAG origin/develop
git merge --no-ff origin/master -m "Merge master ($RELEASE_TAG) into develop"
```
- if merge conflicts occur, solve them and commit
  * `CHANGELOG.md`: keep the release entry and any newer `develop` entries
  * [Version files](#version-files): keep the version from `master`
- check that every [version file](#version-files) has the version of `master`,
  with the check command of that section; fix and commit any file left behind
- push the temporary branch
```bash
git push origin merge/master-to-develop-$RELEASE_TAG
```
- in github create a pull request for `merge/master-to-develop-$RELEASE_TAG` to `develop`
  * one can auto-assign the PR, and doesn't need a review for this PR
  * if `develop` moved meanwhile, "Update branch" is safe on this PR
- after checks complete, please review the checks logs
- do the merge, with merge method "Create a merge commit"
- check that `master` is now contained in `develop`
```bash
git fetch origin
git merge-base --is-ancestor origin/master origin/develop && echo "OK"
```

## cleanup

- delete the (local) release branch and the temporary merge branch
```bash
git checkout develop
git pull --prune
git branch -D release/$RELEASE_TAG
git branch -D merge/master-to-develop-$RELEASE_TAG
git push origin --delete merge/master-to-develop-$RELEASE_TAG
```


# Fed-BioMed hotfix HOWTO

A hotfix is a patch that is added as a new version patch without changing the API or adding new features, when it can't wait for the next release to be corrected. To apply a hotfix, follow these steps:

This procedure makes hotfix of software and documentation (published on `https://docs.fedbiomed.org`).
Both are contained in this repository.

Hotfix principle: follow the [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) hotfix workflow. Can use `git` commands (see example below) or `git-flow` commands

## pre-hotfix

- it is recommended to use a fresh clone of the repository to avoid any side effect

- checkout **to `master` branch** (**Warning: hotfix are not created from `develop` !**), sync your local clone with remote

  ```bash
  git checkout master
  git pull --prune
  ```

- check that the CI for `master` is green: `End-to-End Test`, which runs on
  every push to `master`
- choose a name (eg `521-short-description`) for the issue and assign it to `$HOTFIX_NAME`

  ```bash
  export HOTFIX_NAME=601-short-description
  ```

- set the hotfix version tag for the release (or use this tag directly in commands). For example, if the previous version was `v6.0.0`, it becomes `v6.0.1`.

  ```bash
  export HOTFIX_TAG=v6.0.1
  ```

- fork a `hotfix/$HOTFIX_NAME` branch from `master`, and checkout the `hotfix/$HOTFIX_NAME` branch

  ```bash
  git checkout -b hotfix/$HOTFIX_NAME
  ```

## create hotfix + create version tag

- apply the changes for the hotfix

- in the `hotfix/$HOTFIX_NAME` branch, do the hotfix time updates:
  * `CHANGELOG.md`
  * every file of [Version files](#version-files) that exists on `master`: set
    the new version, then run the check command of that section

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

- create a version tag for the hotfix (on the `hotfix/$HOTFIX_NAME` branch)

  ```bash
  git tag -a $HOTFIX_TAG
  ```
- push the tag to the remote

  ```bash
  git push origin $HOTFIX_TAG
  ```

- check that the documentation pipeline completes successfully
  * new version of documentation is published after a new version tag is pushed. This action builds documentation related contents which are located in `docs/getting-started`, `docs/user-guide`, `docs/developer`, `docs/tutorials`.
  * check that `Build and deploy docs to fedbiomed-docs-public` https://github.com/fedbiomed/fedbiomed/actions/workflows/build-and-deploy-documentation.yml builds correctly
  * review carefully the log details for the build

- check that the docker image publication completes successfully
  * pushing the version tag publishes the `fedbiomed/base`, `fedbiomed/node` and `fedbiomed/researcher` images to Docker Hub
  * check that `Tests and Publish Fed-BioMed Docker Images` https://github.com/fedbiomed/fedbiomed/actions/workflows/docker-deploy.yml completes correctly
  * verify the new version tag is listed on Docker Hub for the three images

- do the merge
  * the merge method MUST be "Create a merge commit" (never "Squash and merge"
    or "Rebase and merge")
  * the tag already published the package, the images and the documentation, so
    the merge publishes nothing. It triggers `End-to-End Test` on `master`,
    which verifies the released tree after the fact.
- if merge conflicts occur, solve them

- browse a few pages in the new documentation on `https://fedbiomed.org` to verify it works as expected

- optionally sync your local clone of `master` with new version of remote

  ```bash
  git checkout master
  git pull -p
  ```

## merge master to develop

`master` (not the hotfix branch) is merged back into `develop`, so that the
merge commit created on `master` is also part of `develop`'s history. This
keeps `master` an ancestor of `develop`, so the next release branch is never
behind `master`.

A direct PR from `master` to `develop` can't be merged because of the
"Require branches to be up to date" rule. Use a temporary branch instead.
**Never click "Update branch" on a PR whose source is `master`**: it would merge
`develop` into `master`.

- create a temporary branch from `develop` and merge `master` into it

  ```bash
  git fetch origin
  git checkout -b merge/master-to-develop-$HOTFIX_TAG origin/develop
  git merge --no-ff origin/master -m "Merge master ($HOTFIX_TAG) into develop"
  ```

- if merge conflicts occur, solve them and commit
  * `CHANGELOG.md`: keep both the hotfix entry and the `develop` entries
  * [Version files](#version-files): keep the version from `master`
- set **every** [version file](#version-files) to the hotfix version, then run
  the check command of that section, and commit. The merge alone is not enough
  when `develop` has version files that `master` does not have (for example the
  `packages/node-api` and `packages/node-ui` versions and pins): they keep the
  old version without any conflict, and the packages can no longer be installed
  together.

  ```bash
  git commit -am "Align package versions to $HOTFIX_TAG"
  ```
- push the temporary branch

  ```bash
  git push origin merge/master-to-develop-$HOTFIX_TAG
  ```

- in github create a pull request for `merge/master-to-develop-$HOTFIX_TAG` to `develop`
  * one can auto-assign the PR, and doesn't need a review for this PR
  * if `develop` moved meanwhile, "Update branch" is safe on this PR
- after checks complete, please review the checks logs
- do the merge, with merge method "Create a merge commit"
- check that `master` is now contained in `develop`

  ```bash
  git fetch origin
  git merge-base --is-ancestor origin/master origin/develop && echo "OK"
  ```

## merge master to an open release branch (only if a release is in progress)

If a `release/*` branch exists when the hotfix is merged into `master`, merge
`master` into it too. Otherwise the release won't contain the hotfix and will
be behind `master`.

```bash
export RELEASE_TAG=v6.1.0   # the release in progress
git fetch origin
git checkout release/$RELEASE_TAG
git pull
git merge --no-ff origin/master -m "Merge master ($HOTFIX_TAG) into release/$RELEASE_TAG"
# conflicts in version files (see "Version files"): keep the release version
# conflicts in CHANGELOG.md: keep both entries
git push origin release/$RELEASE_TAG
```

## cleanup

- delete the (local) hotfix branch and the temporary merge branch

  ```bash
  git checkout develop
  git pull --prune
  git branch -D hotfix/$HOTFIX_NAME
  git branch -D merge/master-to-develop-$HOTFIX_TAG
  git push origin --delete merge/master-to-develop-$HOTFIX_TAG
  ```