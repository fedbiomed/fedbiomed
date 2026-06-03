# Managing Nodes

This page describes the background of how a node process is launched and managed in Fedbiomed. It describes the `NodeProcessManager` class, which is used to start, stop, restart and inspect a node process.

`NodeProcessManager` manages one node subprocess for a single node, which is defined by a single `NodeConfig` class. Its main responsibilities are:

- spawning the node runtime in a child Python process
- terminating that process on request
- logging the process output to a file if the process is launched at background
- storing the current process state and state history in the node database
- verifying the persisted state in the database with the operating-system process table

The source file is `fedbiomed.node.node_pm.py`

## Where It Is Used

The manager is instantiated from CLI in `fedbiomed.node.cli.NodeControl`, when one of the following commands are executed:

- `fedbiomed node start`
- `fedbiomed node start --background`
- `fedbiomed node stop`
- `fedbiomed node restart`
- `fedbiomed node status`

## Workflow

When `NodeProcessManager.start()` is called, it creates a Python child subprocess with the arguments `config` and `node_args`. `Config` is the root path of the node, and `node_args` are the extra arguments passed to the node. 

The argument for the path of config is specified by:

```bash
fedbiomed node -p 'PATH/TO/NODE' start
```

For the node arguments, such as `--background`, you can use `fedbiomed node start --help` for details.

## Public Interface

The methods below are the main methods used for managing the node lifecycle. Nodes can be started and stopped from different terminals, or directly from the same terminal if the option `--background` is used.

| Method | Purpose |
| --- | --- |
| `start(node_args, background=False, actor=None, reason="start_requested")` | Starts the managed node process unless it is already running. |
| `stop(actor=None, reason="stop_requested")` | Terminates the running node process and persists the final state. |
| `restart(node_args, background=False, actor=None, reason="restart_requested")` | Calls `stop()` then `start()` with the same actor and reason. |
| `get_status()` | Returns the current `NodeState`, correcting stale persisted state when possible. |

NodeState is an enum, which can be:

- `NodeState.RUNNING`
- `NodeState.STOPPING`
- `NodeState.STOPPED`
- `NodeState.UNKNOWN`

`Unknown` is returned in when a stored process state does not exist in the database for the node.

When `background=True`, stdout and stderr from the subprocess are appended to:

```text
<node-root>/log/node_process.log
```

When `background=False`, output remains attached to the current terminal and the manager waits for the subprocess to exit.

The optional parameters `actor` and `reason` are explained in more detail in the next section.

## Managing Node Process State

When the user wants to check the node status using the command `fedbiomed node status`, the function `get_status()` is called in the background. `get_status()` does not blindly trust the database. It reads the stored PID and checks it with `psutil`:

- if the process exists but the database is not `RUNNING`, the database is updated to `RUNNING`;
- if the database says `RUNNING` but the process is missing or zombie, the database is updated to `STOPPED`;
- otherwise the stored state is returned.

This check ensures that the CLI and GUI status calls are meaningful; in case of a crash, manual process kill, or stale database entry.

The functions `start()` and `stop()`, update the process state in the database. The updated fields can be seen in the table below:

| Argument   | Description |
| ---        | --- |
| pid: `int`   | The pid of the latest started node process. Normally retrieved from database by default, unless explicitly specified.
| state:  `NodeState`   | The new NodeState after this call. 
| action: `str`   | The action (or function) that triggered the state change. Some default values are: 'start', 'stop' and 'wait'
| actor: `Optional[Dict[str, Any]`]    | Optional information on who initiated the action. By default, if launched from the CLI, it tries to retrieve the local username from the device. By default, if launched from the GUI (or from another web request), it will try to retrieve the user_id, email and role of the user.
| reason: `Optional[str]`   | Optional reason for the state change.
| exit_code: `Optional[int]` | Optional exit code for the process.

Normally the pid and state information shouldn't be given by the user or developer, unless specifically for debugging purposes.

## Database Format

Process state is stored in the node database, under the node root's `var` directory. The manager uses two TinyDB tables:

| Table | Purpose |
| --- | --- |
| `NodeProcessState` | Current state, one entry per `node_id`. |
| `NodeProcessStateHistory` | Append-only history of lifecycle transitions. |

Entries are represented as a `NodeProcessStateEntry` dataclass and include:

- `node_id`
- `node_name`
- `state`
- `action`
- `pid`
- `reason`
- `actor`
- `updated_at`
- `started_at`
- `stopped_at`
- `exit_code`

`updated_at`, `started_at` and `stopped_at` are stored as UTC ISO-8601 strings.
When a manager is initialized, history entries older than 30 days are removed.

## Stop Semantics

`stop()` first persists `STOPPING`, then tries to terminate the process recorded
in the state table:

1. If the manager has no running PID, it logs a warning and returns.
2. If the PID no longer exists, it records `STOPPED` with reason `process_not_found`.
3. Otherwise it sends the signal SIGTERM `terminate()` and waits up to 5 seconds.
4. If the process does not exit, it sends the signal SIGKILL `kill()` and waits up to 5 more seconds.
5. It records `STOPPED` with the exit code if one is available.

The node subprocess also installs `SIGTERM` and `SIGINT` handlers. When the node is connected, those handlers emit a `node_stopped` security event and send an error message to researchers before exiting.

## Foreground Exit Handling

When the node is run in the foreground, the node process is a child of the foreground process, which essentially waits until the node process terminates gracefully. After the node terminates, the foreground process will check the database status, and if the database doesn't have a 'STOPPED' status, it will update it as:

```text
action = "wait"
reason = "process_exited_abruptly"
state = "stopped"
```

This is to double-check the database's integrity in case of an unexpected shutdown in the node process.

## Development Guidance

When extending node lifecycle behavior:

- Please always keep process-control behavior in `NodeProcessManager`, do not add additional functionality into CLI or GUI handlers.
- Use `_set_process_state()` and `_get_process_state()` to read and write process state information.
- Verify `get_status()` returns the correct state, and updates the database if necessary.
- Add targeted unit tests into `tests/test_node_pm.py`.

**See Also:** The [node configuration guide](../user-guide/nodes/configuring-nodes.md), for more user-side information.

Thanks in advance for your contribution to Fed-BioMed! 
