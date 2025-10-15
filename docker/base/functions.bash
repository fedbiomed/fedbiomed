#!/bin/bash

# Global variable to track if a new account was created at runtime
USING_NEW_ACCOUNT=false

# Check if current user is root
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=true
else
    IS_ROOT=false
fi

export IS_ROOT

new_run_time_user() {

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
        echo "info: CONTAINER_GROUP or CONTAINER_GID not set — using default group"
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
        CONTAINER_GROUP=$(id -gn "$FEDBIOMED_USER")
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
    
    # Export CONTAINER_USER and CONTAINER_GROUP for use in other functions
    export CONTAINER_USER
    export CONTAINER_GROUP
}

change_path_owner() {
    local paths=$1

    echo "Checking ownership for container user: $CONTAINER_USER"

    for path in $paths; do
        if [ -e "$path" ]; then
            # Get current owner and group of the path
            current_owner=$(stat -c '%U' "$path" 2>/dev/null)
            current_group=$(stat -c '%G' "$path" 2>/dev/null)
            
            # Check if ownership change is needed
            if [ "$current_owner" != "$CONTAINER_USER" ] || [ "$current_group" != "$CONTAINER_GROUP" ]; then
                echo "info: Current ownership of $path is $current_owner:$current_group, changing to $CONTAINER_USER:$CONTAINER_GROUP"
                
                if chown -R "$CONTAINER_USER:$CONTAINER_GROUP" "$path" 2>/dev/null; then
                    echo "info: Changed ownership of $path (full)"
                else
                    echo "WARNING: Failed to change ownership of $path - continuing anyway"
                    echo "This may happen on cluster environments with restricted permissions"
                    # Try to ensure the directory is at least readable/writable by the user if possible
                    chmod -R u+rw "$path" 2>/dev/null || echo "WARNING: Could not adjust permissions for $path"
                fi
            else
                echo "info: Ownership of $path is already correct ($current_owner:$current_group), skipping"
            fi
        else
            echo "WARNING: Path $path does not exist, skipping ownership change"
        fi
    done
}




