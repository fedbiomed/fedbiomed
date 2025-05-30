#!/bin/bash

new_run_time_user()	{
	 # if using an account/group that does not exist, we need to create it
    # and position a variable to indicate it to the script
    if [ -z "$(getent group $CONTAINER_GID)" -a -z "$(getent group $CONTAINER_GROUP)" ]
    then
        groupadd -g $CONTAINER_GID $CONTAINER_GROUP
        if [ "$?" -ne 0 ]
        then
            echo "CRITICAL: could not create new group CONTAINER_GROUP=$CONTAINER_GROUP CONTAINER_GID=$CONTAINER_GID"
            exit 1
        fi
        echo "info: created new group CONTAINER_GROUP=$CONTAINER_GROUP CONTAINER_GID=$CONTAINER_GID"
        USING_NEW_ACCOUNT=true
    elif [ -z "$(getent group $CONTAINER_GID)" -o -z "$(getent group $CONTAINER_GROUP)" ]
    then
        # case of incoherent group spec : either gid is already used (with another group name)
        # or group name is already used (with another gid)
        echo "CRITICAL: bad group specification, a group already exists with either \
CONTAINER_GROUP=$CONTAINER_GROUP or CONTAINER_GID=$CONTAINER_GID : \
'$(getent group $CONTAINER_GID)' '$(getent group $CONTAINER_GROUP)'"
        exit 1
    fi
    if [ -z "$(getent passwd $CONTAINER_UID)" -a -z "$(getent passwd $CONTAINER_USER)" ]
    then
        # delete stderr to avoid warning messages because homedir already exists
		USERADD_OUTPUT=$(useradd -M -d /home/"$FEDBIOMED_USER" \
			-u "$CONTAINER_UID" -g "$CONTAINER_GID" -s /bin/bash "$CONTAINER_USER" 2>&1)

		if [ "$?" -ne 0 ]; then
			echo "CRITICAL: could not create new user CONTAINER_USER=$CONTAINER_USER CONTAINER_UID=$CONTAINER_UID"
			echo "ERROR: $USERADD_OUTPUT"
			exit 1
		fi
        echo "info: created new user CONTAINER_USER=$CONTAINER_USER CONTAINER_UID=$CONTAINER_UID"
        USING_NEW_ACCOUNT=true

    elif [ -z "$(getent passwd $CONTAINER_UID)" -o -z "$(getent passwd $CONTAINER_USER)" ]
    then
        # case of incoherent user spec : either uid is already used (with another user name)
        # or user name is already used (with another uid)
        echo "CRITICAL: bad user specification, a user already exists with either \
CONTAINER_USER=$CONTAINER_USER or CONTAINER_UID=$CONTAINER_UID : \
'$(getent passwd $CONTAINER_UID)' '$(getent passwd $CONTAINER_USER)'"
        exit 1
    fi

    # If new user was created, add it to the default user's group
    if [ "$USING_NEW_ACCOUNT" = true ]; then
        FEDBIOMED_GROUP=$(id -gn "$FEDBIOMED_USER")
        if getent group "$FEDBIOMED_GROUP" > /dev/null 2>&1; then
            usermod -aG "$FEDBIOMED_GROUP" "$CONTAINER_USER"
            if [ $? -eq 0 ]; then
                echo "info: added $CONTAINER_USER to group $FEDBIOMED_GROUP"
            else
                echo "WARNING: failed to add $CONTAINER_USER to group $FEDBIOMED_GROUP"
            fi
        else
            echo "WARNING: default user group $FEDBIOMED_GROUP not found"
        fi
    fi
}

change_path_owner() {
    local path_nocross=$1  # path to recursively change, dont cross mount boundaries
    local path_full=$2  # path to recursively change, no additional checks

    if [ "$USING_NEW_ACCOUNT" ]
    then
        for path in $path_nocross
        do
            if [ -e $path ]
            then
                # don't use `chown -R` that would cross mountpoints
                find $path -mount -exec chown -h $CONTAINER_USER:$CONTAINER_GROUP {} \;
                if [ "$?" -ne 0 ]
                then
                    echo "CRITICAL: could not change ownership of $path to $CONTAINER_USER:$CONTAINER_GROUP"
                    exit 1
                fi
                echo "info: changed ownership of $path to $CONTAINER_USER:$CONTAINER_GROUP"
            fi
        done

        for path in $path_full
        do
            if [ -e $path ]
            then
                # far quicker than doing a `find`
                chown -R $CONTAINER_USER:$CONTAINER_GROUP $path
                if [ "$?" -ne 0 ]
                then
                    echo "CRITICAL: could not change ownership of $path to $CONTAINER_USER:$CONTAINER_GROUP"
                    exit 1
                fi
                echo "info: changed ownership of $path to $CONTAINER_USER:$CONTAINER_GROUP"
            fi
        done
    else
		# Always keep CONTAINER_USER variable
		CONTAINER_USER=$FEDBIOMED_USER
	fi
}








