**MR description**

TO_BE_FILLED_BY_MR_CREATOR

**Developer Certificate Of Origin (DCO)**

By opening this merge request, you agree the
[Developer Certificate of Origin (DCO)](https://gitlab.inria.fr/fedbiomed/fedbiomed/-/blob/develop/CONTRIBUTING.md#fed-biomed-developer-certificate-of-origin-dco)

This DCO essentially means that:

- you offer the changes under the same license agreement as the project, and
- you have the right to do that,
- you did not steal somebody else’s work.


**Guidelines for MR review**

General:

* give a glance to [DoD](https://fedbiomed.gitlabpages.inria.fr/latest/developer/Fed-BioMed_DoD.pdf)
* check [coding rules and coding style](https://fedbiomed.gitlabpages.inria.fr/latest/developer/usage_and_tools/#coding-style)
* check docstrings (eg run `tests/docstrings/check_docstrings`)

Specific to some cases:

* update all conda envs consistently (`development` and `vpn`, Linux and MacOS)
* if modified researcher (eg new attributes in classes) check if breakpoint needs update (`breakpoint`/`load_breakpoint` in `Experiment()`, `save_state`/`load_state` in aggregators, strategies, secagg, etc.)
