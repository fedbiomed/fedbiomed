# Developer usages and tools

## Introduction

The purpose of this guide is to explicit the coding
rules and conventions used for this project and explain the
use of some of our tools.

This guide is dedicated to all Fed-BioMed developers: Contributors, Reviewers, Team Developers, Core Developers.

Some aspects of this guide may change in the future, stay alert for such changes.


## Code

### Coding environment

Except for some **bash** tools and scripts, the **python** language is used for most parts of the code.

**conda** is used to ease the installation of python and the necessary packages.

### Coding style

We try to stick as close as possible to python coding style as described [here](https://docs.python-guide.org/writing/style/)

We do not enforce coding style validation at each commit. In the future, we may implement some of the tools described [here](https://towardsdatascience.com/4-pre-commit-plugins-to-automate-code-reviewing-and-formatting-in-python-c80c6d2e9f5)

### Coding rules

Project specific [coding rules](https://github.com/fedbiomed/fedbiomed/blob/master/CODING_RULES.md) come in addition to general coding style. Their goal is to favour code homogeneity within the project. They are meant to evolve during the project when needed.

### License

Project code files should begin with these comment lines to help trace their origin:
```
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
```

Code files can be reused from another project with a compatible non-contaminating license.
They shall retain the original license and copyright mentions.
The `CREDIT.md` file and `credit/` directory shall be completed and updated accordingly.

### Authors

Project does not mention authors in the code files. Developers can add themselves to `AUTHORS.md`.


## Repositories

### Framework

The framework is contained in [one git repository](https://github.com/fedbiomed/fedbiomed) with 2 functional parts:

* node: the library and tools to run on each node

* researcher: the library and tools to run on researcher's side

### Documentation

The documentation is contained in the repository under `docs` directory that is used for building [the web site](https://fedbiomed.org). The static files that are obtained after building documentation are kept in the repository `fedbiomed/fedbiomed.github.io` to serve for web.

Fed-BioMed documentation page is configured to be built and published once there is new version tag released.  Publish process is launched as GitHub workflow job where the documentation is built and pushed to public repository `fedbiomed/fedbiomed.github.io`.

#### Events for documentation build

There are two events that trigger documentation publishing:

- `Publish MASTER fedbiomed/fedbiomed.github.io` when pushing a new commit to master

    The documentation website contains static pages such as the home page, about us, and support (main pages). These pages are separate from the documentation versioning process since they can be updated without requiring a new version to be published. As a result, whenever a new commit is pushed to the master branch, the GitHub workflow named `Publish MASTER fedbiomed/fedbiomed.github.io` is triggered. This workflow, located at `.github/workflows/doc-github-io-main-build.yml`, is responsible for publishing the changes made to the main pages.

- `Publish NEW TAG in fedbiomed/fedbiomed.github.io` when pushing a new version tag

    The documentation-related pages located in the directories `getting-started`, `developer`, `tutorials`, and `user-guide` are built whenever a new version tag is pushed. The name of the workflow is `Publish NEW TAG in fedbiomed/fedbiomed.github.io` and the workflow file is located at `.github/workflows/doc-github-io-version-build.yml`.

#### Process flow for documentation deployment

- The workflow file checks out the pushed commit or tag.
- It clones the `fedbiomed/fedbiomed.github.io` repository, which stores all the web static files.
- The documentation is built, and the artifacts are copied into the cloned folder of `fedbiomed/fedbiomed.github.io`.
- Changes are committed and pushed to `fedbiomed/fedbiomed.github.io`.
- The push event triggers the deployment job in the `fedbiomed/fedbiomed.github.io` repository.


## Roles and accesses

Current roles in Fed-BioMed development process are:

- **Fed-BioMed Users**: people using Fed-BioMed for research and/or deployment in collaborative learning applications, and reporting issues.
- **Fed-BioMed Contributors**: developers proposing their changes to the Fed-BioMed code and documentation via pull requests.
- **Fed-BioMed Reviewers**: developers reviewing the pull requests.
    * Reviewers can be Contributors, Team Developers or Core Developers.
- **Fed-BioMed Team Developers**: developers recurrently proposing changes to the Fed-BioMed code and documentation via pull requests, and working in coordinated manner with other Team Developers
    * Currently, Team Developers are chosen by the existing Team Developers among the volunteer Contributors.
- **Fed-BioMed Core Developers**: developers coordinating the coding of components and documentation of Fed-BioMed, design of extensions and modifications the API.
    * Currently, Core Developers also give final approval and merge the pull requests
    * and new Core Developers are chosen by the existing Core Developers among the Team Developers.

In terms of mapping to accounts and roles on GitHub Fed-BioMed repository and organization:

- Users and Contributors have no specific access to the Fed-BioMed repository, they are not member of the Fed-BioMed GitHub organization
-  Reviewers and Team Developers receive the [github repository *write* access](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/repository-roles-for-an-organization).
- Core Developers receive the [github repository *maintain* access](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/repository-roles-for-an-organization).

**Fed-BioMed developers/users access are personal and shall not be shared with someone else.**

Reviewers, Team Developers and Core Developers receive:

- registration as *members* of the [GitHub Fed-BioMed organization](https://github.com/fedbiomed), and membership in the *Developers* team (plus *CoreDevelopers* team for Core Cevelopers) of the organization
- invitation to Fed-BioMed developer Discord server
- registration in Fed-BioMed developer mailing lists (
    discussion list `fedbiomed-developers _at_ inria _dot_ fr`,
    development notifications list `fedbiomed-notifications _at_ inria _dot_ fr`)
- invitation to Fed-BioMed technical team shared files zone on `mybox.inria.fr`

Current list of Core Developers listed by alphabetical order:

- Yannick Bouillard
- Sergen Cansiz
- Francesco Cremonesi
- Marco Lorenzi
- Riccardo Taiello
- Marc Vesin


## Lifecycle

### Gitflow paradigm

The gitflow paradigm must be followed when creating new development branches and for code release ( see [here](https://datasift.github.io/gitflow/IntroducingGitFlow.html) or [here](https://www.atlassian.com/fr/git/tutorials/comparing-workflows/gitflow-workflow))

### Release, next release

Creating a release or integrating a feature to the next release is the responsibility of Core Developers.

As we use the **gitflow** paradigm, the `master` branch of each repository contains the releases.
Next release is integrated under `develop`.

In other words, the `master` and `develop` branches are protected and only writable by Core Developers.

### Pull request

New features are developed in a `feature` branch (refer to gitflow paradigm).

Branch name for developing new features should start with `feature/` and make them easily linkable with the corresponding issue. For example if the branch is related to issue 123, name it `feature/123-my-short-explanation`.

When the feature is ready, the Developer creates a **pull request** (PR) via GitHub. Be sure to request merging to the `develop` branch.

The Core Developers team then assigns the pull request one Core Developer (*Assignee* PR field in GitHub) and one Reviewer (*Reviewer* PR field in GitHub). The *Assignee* and the *Reviewer* can be the same physical person, but they both shall be different people from the developer of the feature.

The *Reviewer* then does a technical review of the pull request evaluating:

- the functional correctness of the feature (eg match with implemented algorithm or formula)
- the maturity of the feature implementation including conformance to the [**definition of done** (DoD)](./definition-of-done.md).
- the absence of technical regression introduced by the feature
- the technical coherence of the implementation of the feature with the existing code base

The *Reviewer* marks the PR as *Approved* in GitHub once it is technically ready to be merged.

The *Assignee* assesses:

- the interest of the feature in relation with the project goal and roadmap
- the absence of functional conflict introduced by the feature
- the valid timeline for merging the feature (if any dependency with other features)

The *Assignee* merges the PR if it meets these requirements and it is *Approved*. If the merging needs to be delayed for some reason, the *Assignee* gives the final approval for merging with its condition/timeline as a comment of the PR.

Once a branch is merged (or stalled , abandoned) it is usually deleted. 
If there is some reason to keep it, it should then be renamed to something starting with `attic/` (eg `attic/short-description-of-branch`).


## Organization and Scrum

* **The development team** is in charge of implementing Fed-BioMed's project goal and roadmap. It carries the bulk of the development effort, coordinating the work of Reviewers, Team Developers, Core Developers.

    It works as an agile team inspiring from [Scrum](https://scrumguides.org/docs/scrumguide/v2020/2020-Scrum-Guide-US.pdf) and loosely implementing it. Development team's work is usually organized in sprints.

    Reviewers, Team Developers and Core Developers are welcome to the team meetings (daily meeting, sprint review, sprint retrospective) in the developer Discord lounge.

    Core Developers are invited to sprint planning meetings.
    Reviewers and Team Developers can be invited to sprint planning meetings depending on their involvement in current actions.

    Participating to the meetings is recommended in periods when one is actively taking part in a major development action where interaction is needed with other team members.

* **External developers** are autonomous developers (Contributors) working at their own pace. This typically fits primarily for punctual contribution, work on some specific function, PoC, etc. External developers are encouraged to interact with the development team to ensure coherence of their planned contributions with the rest of the development activity.


### Product backlog

Product backlog is a [Scrum artifact](https://scrumguides.org/docs/scrumguide/v2020/2020-Scrum-Guide-US.pdf)
composed of the product goal and product backlog entries. Each product backlog entry can contain a functional requirement, a user story, a task, etc.

The current product goal content is:

1. **priority 1** : translating Collaborative Learning to real world healthcare applications
2. **priority 2** : as an open source software initiative, developing of the community, welcoming other contributions 
3. **priority 3** : supporting initiatives that use Fed-BioMed
4. **priority 4** : experimenting new research and technologies

Product backlog entries are:

* all milestones except those with a *[PoC]* mark starting their title
* issues with a *product backlog* label

Product backlog is modified **by the product owner only or with explicit validation of the product owner**.

Modifications of the product backlog include:

- adding new entries (issues/milestones) to the *product backlog*
- during sprint planning, moving issues from the *product backlog* to the new sprint's *sprint backlog* (they are selected for next sprint)
- during sprint planning, moving back uncomplete issues from the previous sprint's *sprint backlog* to the *product backlog* (they won't be continued during next sprint)
- moving *product backlog* issues to *attic* (they are now considered obsolete)
- closing product backlog milestones

Note: product backlog entries and sprint backlog entries can mention "priority 1", etc. in their description to explicitely link to a product goal priority.


### Sprint backlog

Sprint backlog is a [Scrum artifact](https://scrumguides.org/docs/scrumguide/v2020/2020-Scrum-Guide-US.pdf) composed of a sprint goal (why) and product backlog elements selected for the sprint.

Sprint backlog is a plan by and for the developers in order to achieve the sprint goal.


Sprint backlog entries are:

* issues with a *sprint backlog* label

Sprint backlog is created by the development team during the sprint planning. It can be updated and refined during the sprint (new issues, tasks and functional requirements rewriting) in accordance with the sprint goal.

During the sprint planning, all uncomplete entries remaining from the previous sprint's *sprint backlog* can be:

- kept in the *sprint backlog* (they will be continued during next sprint)
- moved back to the *product backlog* (they won't be continued during next sprint)
- moved to the *attic* (they are now considered obsolete) and closed

During the sprint planning, all complete entries from the previous sprint's *sprint backlog*:

- should already be closed (if not, close them) and marked with *done* label
- are removed from the *sprint backlog*


### Proof of concepts

Proof of concept (PoC) are used to experiment new scientific or technical explorations in Fed-BioMed:
PoCs are not bound in time or attached to a sprint. They are closed when they complete or after being stalled for several months.

Proof of concepts are:

* all milestones with a *[PoC]* mark starting their title

PoCs code is not integrated to the next release (no merge). PoCs are not committed to code quality practices (eg: meeting the DoD). When a PoC completes, it may be decided that the PoC functionality:

- will not be implemented: close the PoC milestone
- will be implemented: convert the PoC milestone to a *product backlog* milestone

PoC use branches starting with `poc/` eg `poc/my-short-poc-description`.

PoC is not a Scrum notion.


## Milestones and issues

GitHub milestones and issues are used to keep track of product backlog, sprint backlog and other product items (bugs, proposals, user requests).


### Milestones

Milestones are used to describe mid-term (eg: multi-months) major goals of the project (tasks, functional requirements, user stories).

A milestone is:

* either a proof of concept (PoC)
* or a *product backlog* entry


### Issues

Issues are used to describe smaller goals of the project (tasks, functional requirements, user stories)

An open issue has exactly one type amongst:

* *needs-triage* for new user issues
* *user support*
* a *candidate* 
* a *product backlog* entry
* a *sprint backlog* entry

An issue:

* can be created by a user. It must then be labelled as *needs-triage*
* can be created by an individual developer. It must then label as a *candidate*.
* can be moved from *needs-triage* to *user support* or *candidate* by a team developer. The team developer ensures it contains necessary information and is explicit enough. Team developer can also add *misc* labels.
* can be moved to the *product backlog* by the product owner or with explicit validation of the product owner
* can be moved to the *sprint backlog* during sprint planning by the team developers
* is closed and marked *done* when it is completed. If it belongs to the *sprint backlog*, it should keep this label until the end of the current sprint.

A closed issue has exactly one type amongst:

* *done* (and not anymore in the sprint backlog)
* *attic*

An issue can be labelled as *attic* and closed when it is considered obsolete or not relevant.
It then loses its open issue type label (*needs-triage*, *candidate*, *product backlog*, *sprint backlog*) except for *user support* issues (make it easy to browse past user support issues).


### Labels

Zero or more labels are associated to an issue. 
We sort labels in several categories:

#### *type* labels:

  - **needs-triage** : a user submits a work request to the team (extension proposal, bug, other request)
  - **user support** : a developer validates a user-submitted issue to be a request for support or for information (if it is a bug report or a work request, then it should rather be a labelled as *candidate*)
  - **candidate** : an individual developer submits a work request to the team (extension proposal, bug, other request) or validates a user-submitted issue
  - **product backlog** : the product owner adds an entry to the product backlog
  - **sprint backlog** : the development team adds an entry to the sprint backlog
  - **attic** : the entry is not completed, but is now considered obsolete and closed
  - **done**: the entry is completed, closed, and not anymore in the *sprint backlog*


#### *status* labels:

All sprint backlog issues have one status label. Other issues only have a status label when they are active (eg: a developer not participating to a sprint, a developer working during intersprint).

  - **todo** : issue not started yet (but intention to start soon)
  - **doing** : issue implementation in progress
  - **in review** : issue implementation is finished, a pull request open and is ready for review (or under review)
  - **done** : issue is completed, it meets the DoD and **was merged to the next release integration branch**, but it still belongs to the *sprint backlog*
  

#### *misc* labels

These are optional label that give additional information on an issue.

- **bug** : this issue is about reporting and resolving a suspected bug
- **documentation** : documentation related issue
- **good first issue** : nice to pick for a new contributor

Note: some previously existing tags are now removed - *postponed*, *feature*, *improvement*, *intersprint*


#### Example

* an issue with labels *sprint backlog* + *todo* + *bug* means that this issue is in the current sprint's backlog, that it is not yet started, and that it solves a bug.

* summary :

![Fed-BioMed issues workflow](../../assets/img/fedbiomed_dev_usage_issues.png)


## Other tools

* project file repository (Mybox *Fed-BioMed-tech*)


