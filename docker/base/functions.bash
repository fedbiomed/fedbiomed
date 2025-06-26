#!/bin/bash

new_run_time_user() {

    USING_NEW_ACCOUNT=false

    # Handle group creation -------------------------------------
    if [ -n "$CONTAINER_GID" ] && [ -n "$CONTAINER_GROUP" ]; then
        group_entry_gid=$(getent group "$CONTAINER_GID")
        group_entry_name=$(getent group "$CONTAINER_GROUP")

        if [ -z "$group_entry_gid" ] && [ -z "$group_entry_name" ]; then
            groupadd -g "$CONTAINER_GID" "$CONTAINER_GROUP"
            if [ "$?" -ne 0 ]; then
                echo "CRITICAL: Failed to create group CONTAINER_GROUP=$CONTAINER_GROUP CONTAINER_GID=$CONTAINER_GID"
                exit 1
            fi
            echo "info: Created new group CONTAINER_GROUP=$CONTAINER_GROUP CONTAINER_GID=$CONTAINER_GID"
            USING_NEW_ACCOUNT=true
        elif [ -z "$group_entry_gid" ] || [ -z "$group_entry_name" ]; then
            echo "CRITICAL: Incoherent group specification. One of CONTAINER_GROUP or CONTAINER_GID already exists:"
            echo "  CONTAINER_GID: $(getent group "$CONTAINER_GID")"
            echo "  CONTAINER_GROUP: $(getent group "$CONTAINER_GROUP")"
            exit 1
        fi
    else
        echo "info: CONTAINER_GROUP or CONTAINER_GID not set — skipping group creation"
        CONTAINER_GROUP=$(id -gn "$FEDBIOMED_USER")
    fi

    # Handle user creation --------------------------------
    if [ -n "$CONTAINER_UID" ] && [ -n "$CONTAINER_USER" ]; then
        user_entry_uid=$(getent passwd "$CONTAINER_UID")
        user_entry_name=$(getent passwd "$CONTAINER_USER")

        if [ -z "$user_entry_uid" ] && [ -z "$user_entry_name" ]; then
            USERADD_OUTPUT=$(useradd -M -d /home/"$FEDBIOMED_USER" \
                -u "$CONTAINER_UID" -g "$CONTAINER_GROUP" -s /bin/bash "$CONTAINER_USER" 2>&1)

            if [ "$?" -ne 0 ]; then
                echo "CRITICAL: Failed to create user CONTAINER_USER=$CONTAINER_USER CONTAINER_UID=$CONTAINER_UID"
                echo "ERROR: $USERADD_OUTPUT"
                exit 1
            fi
            echo "info: Created new user CONTAINER_USER=$CONTAINER_USER CONTAINER_UID=$CONTAINER_UID"
            USING_NEW_ACCOUNT=true
        elif [ -z "$user_entry_uid" ] || [ -z "$user_entry_name" ]; then
            echo "CRITICAL: Incoherent user specification:"
            echo "  CONTAINER_UID: $(getent passwd "$CONTAINER_UID")"
            echo "  CONTAINER_USER: $(getent passwd "$CONTAINER_USER")"
            exit 1
        fi
    else
        echo "info: CONTAINER_USER or CONTAINER_UID not set — using default user"
        CONTAINER_USER=$FEDBIOMED_USER
    fi

    # If a new user is created, add it to the default user's group ===
    if [ "$USING_NEW_ACCOUNT" = true ]; then
        FEDBIOMED_GROUP=$(id -gn "$FEDBIOMED_USER")
        if getent group "$FEDBIOMED_GROUP" > /dev/null 2>&1; then
            usermod -aG "$FEDBIOMED_GROUP" "$CONTAINER_USER"
            if [ "$?" -eq 0 ]; then
                echo "info: Added $CONTAINER_USER to group $FEDBIOMED_GROUP"
            else
                echo "WARNING: Failed to add $CONTAINER_USER to group $FEDBIOMED_GROUP"
            fi
        else
            echo "WARNING: Default user group $FEDBIOMED_GROUP not found"
        fi
    fi
}

change_path_owner() {
    local path_nocross=$1
    local path_full=$2

    if [ "$USING_NEW_ACCOUNT" = true ]; then
        echo "Using run-time defined user: $CONTAINER_USER"

        for path in $path_nocross; do
            if [ -e "$path" ]; then
                find "$path" -mount -exec chown -h "$CONTAINER_USER:$CONTAINER_GROUP" {} \; || {
                    echo "CRITICAL: Failed to change ownership of $path"
                    exit 1
                }
                echo "info: Changed ownership of $path (no-cross)"
            fi
        done

        for path in $path_full; do
            if [ -e "$path" ]; then
                chown -R "$CONTAINER_USER:$CONTAINER_GROUP" "$path" || {
                    echo "CRITICAL: Failed to change ownership of $path"
                    exit 1
                }
                echo "info: Changed ownership of $path (full)"
            fi
        done
    else
        CONTAINER_USER=$FEDBIOMED_USER
    fi
}




