# Managed Runs

An agent that uses `caliper-mcp` can start a Python script of the workspace, watch it, fork it with a changed parameter and stop it. It needs no terminal and no script of its own. The model runner of the app starts each script as a child process, which this page calls a managed run. Five tools give the agent this control. [Run Control and Events](run-control-and-events.md) describes the tools that then act on a run.

## Tools

Each command is one MCP tool with the first dot of its name replaced by `_`, such as `run_start`. The commands run on the frame thread of the app. A command never waits for a child process. Thus `run.start` and `run.fork` answer at once with the state `starting`, and the runner finishes the start in the background.

| Command | Arguments | Function |
|---|---|---|
| `run.list` | None | Lists the managed runs and the script that you attached yourself |
| `run.start` | `script`, `args`, `params`, `id` | Starts a script of the workspace as a managed run |
| `run.fork` | `run`, `params`, `id` | Starts a new run from a checkpoint of a run, with changed parameters |
| `run.stop` | `run`, `kill_after_s` | Stops a run at its next checkpoint, and ends its process after a limit |
| `run.logs` | `run`, `lines` | Returns the last lines of the output of a run |

The `script` is a path in the workspace, relative or absolute. The command fails with `run.script_refused` when the real path of the file is outside the workspace. The `params` argument is a list of at most 64 pairs of `name` and `value`, because every schema of the command table is closed. Without `id`, the runner makes the id from the file name of the script and a number, such as `train-1`. An id must match `[a-z0-9-]{1,40}`.

## Start and State

A run goes through these states: `starting`, `running`, `paused`, `exited` and `failed`. The runner sends a `run` event of change `runner` each time a run changes its state. The agent waits for that event with `events.wait`, or it reads `run.list`. A run is `running` when its script has written its stream link and the runner has connected. A run that ends with the exit code 0, or that ends after a stop, is `exited`. A run that ends with another code, or that writes no link file within 120 seconds, is `failed`.

Each entry of `run.list` has `id`, `managed`, `state`, `script`, `args`, `params`, `parent`, `step`, `exit_code`, `attached`, `stopping` and `error`. The script that you attached yourself, with `--url` or from the landing screen, shows with the id `user` and `managed` false.

## Fork

`run.fork` takes a managed run that runs or that is paused. The runner asks the parent for a checkpoint. Then it starts the same script with the same arguments, resumes it from the checkpoint file and applies the new `params`. The fork keeps the model and optimizer state of the parent, and every parameter that the fork does not name keeps the value that the parent had. The fork records the id of its parent in `parent`. If the checkpoint fails, the fork goes to the state `failed` with the message of the action.

## Stop

`run.stop` asks the script to end at its next checkpoint, and the script ends with the exit code 0. When the process is still alive after `kill_after_s` seconds, the runner asks it to end by its process id. After 2 more seconds, the runner ends it by force. The default limit is 10 seconds. A run without a connection yet ends at once. No process outlives the app, because the runner ends every child by its process id when the app closes.

## Review Cards

A managed run executes Python code on your computer. As a result, `run.start` and `run.fork` always wait for your card, in every mode. No rule of a session and no saved rule removes this card, and the mode Allow all does not remove it. The mode Read only refuses both commands. The card shows the script, the arguments and the parameters. `run.stop` is a `run` command as in [Run Control and Events](run-control-and-events.md#approval-rules): it asks the origins `agent` and `mcp`, and a rule can allow it.

## Logs and Files

The runner writes the standard output and the standard error of each run into `output.log` of its folder, `runs/<id>/` in the workspace. The same folder holds the link file and the checkpoint files. `run.logs` returns the last lines of the log, from at most the last 256 KiB of the file. The default is 50 lines, and the most is 200.

## Two Runs in One Panel

The app attaches to one run at a time. It attaches to a new run when that run becomes ready. The commands `run.state`, `run.set_param`, `run.action` and `run.pause` take an optional `run` argument. When it names a run that the app does not show, the app switches its attachment to that run at the next frame.

A panel can draw the data of a run that the app does not show. The catalog source `run:<id>/<view>` names the series `<view>` of the managed run `<id>`, such as `run:train-1/loss`. The runner reads it through its own connection to the run, and it keeps the last 262,144 values of each series until the app closes. A `series` panel draws it as it draws `live:loss`. The series of a fork start at the step of its checkpoint, but `live:loss` of the run that the app shows counts its values from 0. Thus one panel with the sources `run:train-1/loss` and `run:train-2/loss` shows a run and its fork on one step axis, where the fork branches from its parent. `catalog.list` with the kind `run` shows these sources.

## Errors

| Code | Cause |
|---|---|
| `command.not_available` | The session has no model runner |
| `run.script_refused` | The script is outside the workspace, or it is not a file |
| `run.bad_id` | The id does not match `[a-z0-9-]{1,40}` |
| `run.exists` | A run of this id exists in the workspace |
| `run.not_found` | No run of this id |
| `run.not_ready` | The run has no connection yet, or it ended |
| `run.not_managed` | The command needs a managed run, and `user` is your own script |
| `run.start_failed` | The interpreter or the process could not start |
