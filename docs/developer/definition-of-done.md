# Definition of Done for Fed-BioMed

v1.1 - 2023-05-31


The Definition of Done is a set of items that must be completed and quality measures that must be met, before a task or a user story can be considered complete. The DoD gives the team a shared understanding of the work that was completed. 

## Validate CI

- Pass CI build tests 
- Make sure documentation test build process is passed. Changes in docstring and documentation impacts documentation build process.  


## Review of the code 

The reviewer can question any aspect of the increment in coherence with [Usage and Tools](./usage_and_tools.md#merge-request), exchange with the developer (good practice : leave a github trace of the exchanges), approve it or not.

- Be specific in the pull request about what to review (critical code or properties of the code).
- Coding style: inspire from PEP-8.
- Understand the code and try to detect bugs.
- Remove detected bugs.

## Documentation

### Code 
- Comment critical or difficult points in the code; obvious lines (eg: tests) are excluded from comments.
- Add `FIXME` or `TODO` tags for any detected bugs or improvements that are technically beyond the scope of the pull request.
- Write minimal comments in the code (docstring) for a function or a class: parameters and typing, return, purpose of the class or the function.


### User/Developer Docs
- Add API reference in `docs/developer/api` if there is a new module introduced.
- Write minimal documentation for scripts (separate README file) or notebooks (inside the notebook).
- Update/Add documentation in `docs` if there is a new feature or change in API that impacts the content or examples in documentation.

## Write unit-test for the code

- Be clever : put reasonable effort on writing tests. Current target of unit tests is to reach 100% coverage of code, with reasonably clever functional coverage.
- Add unit test when correcting a bug.

Please refer to [the guide of unit testing practices](./testing-in-fedbiomed.md) before starting to write or to modify the unit tests in Fed-BioMed. 

## Post-merge actions

After merging:

- close the pull request
- update the issue
- if the pull request terminates the issue, mark the issue as `done` and close it.
