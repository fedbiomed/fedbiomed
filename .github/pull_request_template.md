
**MR description**

TO_BE_FILLED_BY_MR_CREATOR

**Developer Certificate Of Origin (DCO)**

By opening this merge request, you agree to the
[Developer Certificate of Origin (DCO)](https://github.com/fedbiomed/test-fedbiomed/blob/develop/CONTRIBUTING.md#fed-biomed-developer-certificate-of-origin-dco)

This DCO essentially means that:

- you offer the changes under the same license agreement as the project, and
- you have the right to do that,
- you did not steal somebody else’s work.

**License**

Project code files should begin with these comment lines to help trace their origin:
```
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
```

Code files can be reused from another project with a compatible non-contaminating license.
They shall retain the original license and copyright mentions.
The `CREDIT.md` file and `credit/` directory shall be completed and updated accordingly.


**Guidelines for MR review**

General:

* give a glance to [DoD](https://fedbiomed.org/latest/developer/Fed-BioMed_DoD.pdf)
* check [coding rules and coding style](https://fedbiomed.org/latest/developer/usage_and_tools/#coding-style)
* check docstrings (eg run `tests/docstrings/check_docstrings`)

Specific to some cases:

* update all conda envs consistently (`development` and `vpn`, Linux and MacOS)
* if modified researcher (eg new attributes in classes) check if breakpoint needs update (`breakpoint`/`load_breakpoint` in `Experiment()`, `save_state`/`load_state` in aggregators, strategies, secagg, etc.)
* if modified a component with versioning (config files, breakpoint, messaging protocol) then update the version following the rules in `common/utils/_versions.py`