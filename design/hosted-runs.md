# Hosted Runs

This page defines the app side of milestone M4, Hosted Runs. The model runner of the app starts a Python script of the workspace as a managed run, attaches to it, forks it from a checkpoint and stops it. An agent drives the runner through five commands of `caliper-mcp`, and it needs no terminal and no script of its own. The script side, `caliper.watch()` with its checkpoint and stop actions, is part of Caliper LIB and follows the same process contract.

## Status

The commands, the review card, the runner and the run sources are in code. The process code for Windows compiles and is not tested. The acceptance test of M4 and the docs pages come after the two tracks of M4 merge.

| File | Content |
|---|---|
| `src/builder/include/caliper_builder/session/runner_commands.h` | The `ModelRunner` interface of the session and the command handlers |
| `src/builder/session/runner_commands.cpp` | `run.list`, `run.start`, `run.stop`, `run.fork` and `run.logs` |
| `src/host/runner/child_process.h` | The child process interface, with `child_process_posix.cpp` and `child_process_win.cpp` |
| `src/host/runner/run_link.h` | The runner's own viewer connection to one managed run |
| `src/host/runner/model_runner.h` | The app's model runner, its run table and its run sources |

## Process Contract

A managed run is a Python process. The runner starts the workspace's interpreter on the script, with the workspace folder as the working folder. The child gets the app's environment without the variables that start with `CALIPER_RUN_`, then the variables of the contract and `PYTHONUNBUFFERED=1`.

| Variable | Value |
|---|---|
| `CALIPER_RUN_ID` | The run's id, `[a-z0-9-]{1,40}` |
| `CALIPER_RUN_DIR` | The folder `<workspace>/runs/<id>/`, which the runner makes |
| `CALIPER_RUN_LINK_FILE` | `<run dir>/link.txt`, where the script writes its stream link |
| `CALIPER_RUN_PARAMS` | The `params` argument as a JSON object of values by name, when it has values |
| `CALIPER_RUN_RESUME` | The checkpoint file of the parent, for a fork |
| `CALIPER_RUN_PARENT` | The id of the parent, for a fork |

The interpreter is `CALIPER_RUN_PYTHON` when that variable is set. Else it is the interpreter of the workspace's `.venv` folder, when one exists, and else `python3` on the `PATH`. The runner writes the standard output and the standard error of the child into `<run dir>/output.log`. The exit of the process is the end of the run, and the runner records the exit code.

## Commands

The five commands join the command table. Each one becomes an MCP tool with the dot replaced by `_`, such as `run_start`.

| Command | Effect | Function |
|---|---|---|
| `run.list` | `read` | Lists the managed runs and the person's attached script |
| `run.start` | `run` | Starts a script of the workspace as a managed run |
| `run.fork` | `run` | Starts a new run from the next checkpoint of a run, with changed parameters |
| `run.stop` | `run` | Stops a managed run at its next checkpoint, and ends its process after a limit |
| `run.logs` | `read` | Returns the last lines of the output of a managed run |

The commands run on the frame thread, as the run commands of M1 do. A handler never waits for a child process. Thus `run.start` and `run.fork` answer when the process starts, with the state `starting`. The runner then waits for the link file, connects, attaches the app and sends a `run` event of change `runner`. The agent waits for that event with `events.wait`, or it reads `run.list`.

### Run Start

`run.start` takes `script`, and optionally `args`, `params` and `id`. The script is a file path, relative to the workspace or absolute. Its real path must be in the workspace, else the command fails with `run.script_refused`. The arguments are at most 64 strings. The parameters are a list of at most 64 pairs of `name` and `value`, and a value is a number, a Boolean or a string. The list is not an object, because each schema of the command table is closed. Without `id`, the runner makes the id from the file name of the script and a number, such as `train-1`. An id whose run folder exists already fails with `run.exists`.

### Run Fork

`run.fork` takes `run`, and optionally `params` and `id`. The parent must be a managed run that runs or that is paused, else the command fails with `run.not_ready`. The runner sends the action `caliper.checkpoint` to the parent through its own connection. When the action returns the path of the checkpoint, the runner starts the same script with the same arguments, with `CALIPER_RUN_RESUME`, `CALIPER_RUN_PARENT` and the new parameters. `CALIPER_RUN_PARAMS` of a fork holds only the `params` of the fork, because the checkpoint holds the current values of the parent. When the action fails, the fork goes to the state `failed` with the message of the action.

### Run Stop

`run.stop` takes `run`, and optionally `kill_after_s`, from 0 to 600 seconds, with a default of 10. The runner sends the action `caliper.stop` to the run. When the process has not ended after the limit, the runner ends it by its process id: first with a request to end, then with a forced end after 2 seconds. A run that has no connection yet ends at once. The result holds the run with `stopping` true.

### Run List

`run.list` takes no arguments. It returns `runs`, the managed runs in the order of their start. When the app is attached to a script that the runner did not start, the list also holds that script with the id `user` and `managed` false.

| Key | Meaning |
|---|---|
| `id` | The run's id |
| `managed` | Whether the runner started the run |
| `state` | `starting`, `running`, `paused`, `exited` or `failed` |
| `script` | The script, relative to the workspace for a managed run |
| `args`, `params` | The arguments and the parameters of the start |
| `parent` | The id of the parent of a fork, else null |
| `step` | The step of the latest checkpoint, else null |
| `exit_code` | The exit code, else null. A signal gives its number, negative. |
| `attached` | Whether the app is attached to the run |
| `stopping` | Whether a stop was asked |
| `error` | Why the run failed, else null |

A run that ends with the exit code 0, or that ends after a stop, is `exited`. A run that ends with another code, that writes no link file before the start limit of 120 seconds, or whose fork found no checkpoint, is `failed`.

### Run Logs

`run.logs` takes `run`, and optionally `lines`, from 1 to 200, with a default of 50. It returns the last lines of `output.log`, from at most the last 256 KiB of the file.

## Review Card

A managed run executes Python code, thus `run.start` and `run.fork` always wait for a card, as a Python loader does. The card shows for every origin except `ui` and `builder`, and in every mode. No rule of a session and no saved rule removes it, and Allow all does not either. The mode Read only refuses the two commands. `run.stop` is a `run` command as in M1: it asks the origins `agent` and `mcp`, and a rule can allow it.

| Command | Card Text |
|---|---|
| `run.start` | Start `<script>` as the managed run `<id>`, with the arguments `<args>` and the parameters `<params>`. The script runs as Python code on this computer. |
| `run.fork` | Fork the run `<run>` at its next checkpoint, with the parameters `<params>`. The fork runs `<script>` as Python code beside its parent. |
| `run.stop` | Stop the run `<run>` at its next checkpoint, and end its process after `<n>` seconds. |

## Model Runner

The model runner is a host component of the app in `src/host/runner/`. It owns the child processes, one viewer connection for each managed run and the table of runs. The session reaches it through the interface `ModelRunner` of `runner_commands.h`, which the app gives to `Session::set_model_runner()`. A session without a runner answers the five commands with `command.not_available`, as it does for the run commands without a run control.

### Child Processes

The interface `ChildProcess` of `child_process.h` starts a program with arguments, environment, working folder and an output file. It reads the exit without a wait, and it ends the process by its id. The POSIX code uses `posix_spawn` and puts the child in a process group of its own, so that a terminal signal to the app does not reach the child. The Windows code uses `CreateProcessW` and a job object that ends the child when the app closes. The Windows code is not tested.

No process outlives the app. When the app closes, the runner ends every child by its process id and waits for each exit, at most 2 seconds before the forced end.

### Runner Connection

For each managed run, the runner opens its own viewer connection to the link in the link file, with the capability `controls`. It reads the run state and the step from CONTROLS, it sends the actions `caliper.checkpoint` and `caliper.stop`, and it subscribes to each series view of the run. This connection is separate from the app's attachment. Thus the runner reads a run that the app does not show, and the app's one attachment does not change when the runner acts.

### Attachment

The app attaches to one run at a time. When a managed run becomes ready, the runner attaches the app to it at the next frame, as the option `--url` does. The run commands of M1 (`run.state`, `run.set_param`, `run.action`, `run.pause`) take an optional argument `run`, a managed run's id or `user`. A request for a managed run that the app does not show goes through the runner's connection, and the app switches its attachment to that run at the next frame.

## Run Sources

The catalog kind `run` names the series of the managed runs. The id `run:<id>/<view>` names the series `<view>` of the run `<id>`, such as `run:train-1/loss`. The runner keeps the last 262,144 values of each series of each run, also after the run ends, until the app closes. A `series` panel draws a run source as it draws `live:loss`. Thus two panels, or one panel with two `y` slots, show the loss of a run and of its fork side by side.

## Runner Events

The runner sends a `run` event of change `runner` each time a managed run changes its state. The event has the keys `run`, `state`, `parent`, `exit_code` and `error`.

## Errors

| Code | Cause |
|---|---|
| `command.not_available` | The session has no model runner |
| `run.script_refused` | The script is outside the workspace, or it is not a file |
| `run.bad_id` | The id does not match `[a-z0-9-]{1,40}` |
| `run.exists` | A run of this id exists in the workspace |
| `run.not_found` | No run of this id |
| `run.not_ready` | The run has no connection yet, or it ended |
| `run.not_managed` | The command needs a managed run, and `user` is the person's script |
| `run.start_failed` | The interpreter or the process could not start |

## Open Points

- When the app ends without its normal close, such as in a crash, the POSIX children stay. The script side can end itself when its parent process goes. The Windows job object covers this case on Windows.
- A run source keeps its values only in memory. A later milestone can store them in the run database (RUN-03).
