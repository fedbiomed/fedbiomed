#!/bin/bash
set -x

# Permission for writing files into fedbiomed directory
if [ -n "$NB_UID" ] ; then
    find /home/fed/ -mount -exec mountpoint {} >/dev/null \; -or -exec chown $NB_UID {} \;
fi


# Configure workspace
WORKSPACE='/home/workspace'
[ -n "$NB_UID" ] && chown $NB_UID $WORKSPACE
[ -n "$NB_GID" ] && chgrp $NB_GID $WORKSPACE

ln -s $WORKSPACE /home/jovyan/


