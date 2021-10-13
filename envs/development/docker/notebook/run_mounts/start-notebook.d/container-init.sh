#!/bin/bash
set -x

# Permission for writing files into fedbiomed directory
if [ -n "$NB_UID" ] ; then
    find /home/fed/ -mount -exec mountpoint {} >/dev/null \; -or -exec chown $NB_UID {} \;
fi



NOTEBOOKPATH='/home/workspace'

[ -n "$NB_UID" ] && chown $NB_UID $NOTEBOOKPATH
[ -n "$NB_GID" ] && chgrp $NB_GID $NOTEBOOKPATH

[ -e "$NOTEBOOKNAME" ] || ln -s $NOTEBOOKPATH /home/jovyan/$NOTEBOOKNAME

# if [ -n "$NB_UID" ] ; then
#     find /home/workspace -mount -exec mountpoint {} >/dev/null \; -or -exec chown $NB_UID {} \;
# fi

# if [ -n "$NB_GID" ] ; then
#     find /home/workspace -mount -exec mountpoint {} >/dev/null \; -or -exec chgrp $NB_UID {} \;
# fi

