#!/bin/bash
set -e
set -u
set -o pipefail

 ls -la .


# Clean unnecessary files after installation
# rm -rf ~/.cache/pip
# rm -rf docs/tutorials
# rm -rf notebooks
# rm -rf envs/vpn
# rm -rf envs/common
# rm -rf envs/common_reference
# rm -rf fedbiomed_gui

readonly FEDBIOMED_USER="${FEDBIOMED_USER:-fedbiomed}"
readonly CONTAINER_UID="${CONTAINER_UID:-${FEDBIOMED_UID}}"
readonly CONTAINER_GID="${CONTAINER_GID:-${FEDBIOMED_GID}}"

# Logging helper
log() {
    echo "[entrypoint] $*" >&2
}


# Validate UID/GID ranges
# Validate UID/GID ranges
validate_ids() {
    local uid=$1
    local gid=$2
    
    # Only block root (0)
    if [ "$uid" -eq 0 ]; then
        log "ERROR: Cannot use UID 0 (root)"
        return 1
    fi
    
    if [ "$gid" -eq 0 ]; then
        log "ERROR: Cannot use GID 0 (root group)"
        return 1
    fi
    
    # Validate upper bounds
    if [ "$uid" -gt 65535 ] || [ "$gid" -gt 65535 ]; then
        log "ERROR: UID/GID must be <= 65535 (got UID=$uid, GID=$gid)"
        return 1
    fi
    
    return 0
}

# Only perform root operations if running as root
if [ "$(id -u)" = "0" ]; then
    log "Running as root - performing initialization..."
    
    # Validate inputs
    if ! validate_ids "$CONTAINER_UID" "$CONTAINER_GID"; then
        exit 1
    fi
    
    # Verify user exists
    if ! id "$FEDBIOMED_USER" >/dev/null 2>&1; then
        log "ERROR: User $FEDBIOMED_USER does not exist"
        exit 1
    fi
    
    # Get current UID/GID
    CURRENT_UID=$(id -u "$FEDBIOMED_USER")
    CURRENT_GID=$(id -g "$FEDBIOMED_USER")
    
    # Adjust UID/GID if provided and different from current
    if [ "$CURRENT_UID" != "$CONTAINER_UID" ] || [ "$CURRENT_GID" != "$CONTAINER_GID" ]; then
        log "Adjusting user permissions to UID:GID = $CONTAINER_UID:$CONTAINER_GID"
        
        # Modify group first (usermod requires group to exist)
        if [ "$CURRENT_GID" != "$CONTAINER_GID" ]; then
            log "Changing GID from $CURRENT_GID to $CONTAINER_GID"
            groupmod -g "$CONTAINER_GID" "$FEDBIOMED_USER" 2>/dev/null || {
                log "WARNING: Failed to change GID, continuing..."
            }
        fi
        
        # Modify user UID
        if [ "$CURRENT_UID" != "$CONTAINER_UID" ]; then
            log "Changing UID from $CURRENT_UID to $CONTAINER_UID. This may take a moment if there are many files to update...."
            usermod -o -u "$CONTAINER_UID" "$FEDBIOMED_USER" 2>/dev/null || {
                log "WARNING: Failed to change UID, continuing..."
            }
        fi
        
        # Fix ownership of home directory -------------------------------------------------------------------
        log "Fixing ownership of /home/$FEDBIOMED_USER"
        chown -R "$CONTAINER_UID:$CONTAINER_GID" "/home/$FEDBIOMED_USER" 2>/dev/null || {
            log "WARNING: Failed to fix ownership of some files in home directory"
        }

        # Fix supervisord directories if they exist (for node containers) -----------------------------------
        if [ -d "/var/log/supervisor" ]; then
            log "Fixing ownership of /var/log/supervisor"
            chown -R "$CONTAINER_UID:$CONTAINER_GID" "/var/log/supervisor" 2>/dev/null || true
        fi

        if [ -d "/var/run/supervisor" ]; then
            log "Fixing ownership of /var/run/supervisor"
            chown -R "$CONTAINER_UID:$CONTAINER_GID" "/var/run/supervisor" 2>/dev/null || true
        fi

        if [ -f "/etc/supervisor/supervisord.conf" ]; then
            log "Fixing ownership of /etc/supervisor/supervisord.conf"
            chown "$CONTAINER_UID:$CONTAINER_GID" "/etc/supervisor/supervisord.conf" 2>/dev/null || true
        fi

        # Fix ownership of /fbm-node if it exists
        if [ -d "/fbm-node" ]; then
            log "Fixing ownership of /fbm-node"
            chown -R "$CONTAINER_UID:$CONTAINER_GID" "/fbm-node" 2>/dev/null || {
                log "WARNING: Failed to fix ownership of /fbm-node"
            }
        fi
        
        # Fix ownership of /fbm-researcher if it exists
        if [ -d "/fbm-researcher" ]; then
            log "Fixing ownership of /fbm-researcher"
            chown -R "$CONTAINER_UID:$CONTAINER_GID" "/fbm-researcher" 2>/dev/null || {
                log "WARNING: Failed to fix ownership of /fbm-researcher"
            }
        fi
    else
        log "User $FEDBIOMED_USER already has correct UID:GID ($CONTAINER_UID:$CONTAINER_GID)"
    fi
    
    # Execute root-level component setup if exists (e.g., nginx config for node-gui)
    if [ -f /entrypoint-root.sh ]; then
        log "Executing /entrypoint-root.sh as root"
        bash /entrypoint-root.sh || {
            log "ERROR: /entrypoint-root.sh failed"
            exit 1
        }
    fi
    
    log "Root operations complete - dropping privileges to $FEDBIOMED_USER"
    
    # DROP PRIVILEGES and execute entrypoint or keep running
    if [ -f /entrypoint.sh ]; then
        exec su -s /bin/bash "$FEDBIOMED_USER" -c "exec /entrypoint.sh $*"
    else
        log "No /entrypoint.sh found - keeping container alive"
        exec su -s /bin/bash "$FEDBIOMED_USER" -c "exec sleep infinity"
    fi
else
    log "Already running as non-root ($(id -un):$(id -u):$(id -g))"
    log "Skipping UID/GID adjustment"
    
    # Already non-root, go directly to entrypoint or keep running
    if [ -f /entrypoint.sh ]; then
        exec /entrypoint.sh "$@"
    else
        log "No /entrypoint.sh found"
        log "Container is running as: $(id -un):$(id -u):$(id -g)"
        exec sleep infinity
    fi
fi

# Should never reach here
log "ERROR: Failed to execute user entrypoint"
exit 1
