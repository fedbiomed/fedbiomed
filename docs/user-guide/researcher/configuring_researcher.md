## Creating advanced configuration files for `researcher` components

The script `${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration create -c RESEARCHER -n [CONFIGURATION_FILE_NAME]` will create or optionally recreate (`-f`) a configuration file for a node with `CONFIGURATION_FILE_NAME` name in the `${FEDBIOMED_DIR}/etc` folder.
More options are available and described through the help menu: `${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration create -h`
The parametrization of this script with regard to the various fields stored in the configuration happens through the usage of environment variables.
The fields that can be controlled, their associated evironment variable and default value are described as follow:

[server]:
- host: RESEARCHER\_SERVER\_HOST, ${IP\_ADDRESS}, if not set: localhost
- port: RESEARCHER\_SERVER_PORT, 50051

### Examples:
```
$ RESEARCHER_SERVER_HOST=121.203.21.147 RESEARCHER_SERVER_PORT=8909 ${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration create -c RESEARCHER -n researcher.ini -f
```

Note that recreating (`-f`) a configuration file will override the whole file including the node id. 
To preserve the node id across updates of the configuration file, prefer using `${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration refresh -c RESEARCHER -n researcher.ini`.
