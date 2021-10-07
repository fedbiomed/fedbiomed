#!/bin/bash
set -x

# add commands to be run before options (group, user, etc.) are applied

# not properly handled by container init script CHOWN options
if [ -n "$NB_USER" ] ; then
    ln -s /home/jovyan "/home/$NB_USER"
fi
if [ -n "$NB_UID" ] ; then
    find /home/jovyan/ -mount -exec mountpoint {} >/dev/null \; -or -exec chown $NB_UID {} \;
fi
if [ -n "$NB_GID" ] ; then
    find /home/jovyan/ -mount -exec mountpoint {} >/dev/null \; -or -exec chgrp $NB_UID {} \;
fi


# glitch: make a copy of sample notebooks - owned and writable by the container,
# saved outside of container, not git'ed
NOTEBOOKNAME='notebook'
NOTEBOOKPATH="/home/$NOTEBOOKNAME/$NOTEBOOKNAME"
if [ ! -d "$NOTEBOOKPATH" ] ; then
    mkdir $NOTEBOOKPATH
    [ -n "$NB_UID" ] && chown $NB_UID $NOTEBOOKPATH
    [ -n "$NB_GID" ] && chgrp $NB_GID $NOTEBOOKPATH
fi
[ -e "/home/jovyan/$NOTEBOOKNAME" ] || ln -s $NOTEBOOKPATH /home/jovyan/$NOTEBOOKNAME

for file in $(cd /home/samples ; ls -1) ; do
    [ -e "$NOTEBOOKPATH/$file" ] || cp -p /home/samples/$file $NOTEBOOKPATH/$file
done

if [ -n "$NB_UID" ] ; then
    find $NOTEBOOKPATH/ -mount -exec mountpoint {} >/dev/null \; -or -exec chown $NB_UID {} \;
fi
if [ -n "$NB_GID" ] ; then
    find $NOTEBOOKPATH/ -mount -exec mountpoint {} >/dev/null \; -or -exec chgrp $NB_UID {} \;
fi
