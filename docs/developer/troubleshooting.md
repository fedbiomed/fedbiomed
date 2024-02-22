# Troubleshooting 

## 1. Error while installing MP-SDPZ through `fedbiomed_configure_secagg` on MacOS M3

While installing MP-SPDZ using the script `fedbiomed_configure_secagg` you may get error due to unmaintained `mpir` module. To solve this problem please follow the steps below.

1 - Execute `brew edit mpir`. In the editor find the line that starts with 'disable! date: "xxxx", because: unmaintaned' and comment it. Save it and exit `:wq`. 
2 - Execute `{FEDBIOMED_DIR}/scripts/fedbiomed_configure_secagg` by environment variable `HOMEBREW_NO_INSTALL_FROM_API=1`.

```shell
HOMEBREW_NO_INSTALL_FROM_API=1 {FEDBIOMED_DIR}/scripts/fedbiomed_configure_secagg
```
