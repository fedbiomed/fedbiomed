**Guidelines for MR review**

General:

* give a glance to [DoD](https://fedbiomed.gitlabpages.inria.fr/latest/developer/Fed-BioMed_DoD.pdf)
* check [coding rules and coding style](https://fedbiomed.gitlabpages.inria.fr/latest/developer/usage_and_tools/#coding-style)
* check docstrings (eg run `tests/docstrings/check_docstrings`)

Specific to some cases:

* update all conda envs consistently (`development` and `vpn`, Linux and MacOS)
* if modified researcher (eg new attributes in classes) check if breakpoint needs update (`breakpoint`/`load_breakpoint` in `Experiment()`, `save_state`/`load_state` in aggregators, strategies, secagg, etc.)
