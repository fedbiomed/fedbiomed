#!/bin/bash
set -x


# Permission for writing files into fedbiomed directory
if [ -n "$NB_UID" ] ; then
    find /home/fed/ -mount -exec mountpoint {} >/dev/null \; -or -exec chown $NB_UID {} \;
fi

