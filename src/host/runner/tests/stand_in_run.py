"""A stand-in of a managed run for the model runner's tests (design/hosted-runs.md, Process Contract).

It follows the process contract of a managed run with the public Caliper LIB only, so that the runner
can be tested before caliper.watch() implements the contract: it serves its stream, writes the link
file atomically, applies CALIPER_RUN_PARAMS to the parameter lr, resumes from CALIPER_RUN_RESUME,
and registers the actions caliper.checkpoint (writes checkpoint-<step>.json and returns its path) and
caliper.stop (ends the script with exit code 0 at its next step). Each step publishes the series
loss, which falls by the factor (1 - lr) at each step, and prints one line.

Arguments: the number of steps (default: run until stopped), then the seconds of each step, then
optionally exit-after-checkpoint, which ends the process with exit code 0 at once after the step
whose checkpoint it saved, as soon as the reply is out, so that the runner can see the reply and the
exit in one pass.
"""
import json
import os
import sys
import time

import caliper


def main() -> None:
    run_dir = os.environ["CALIPER_RUN_DIR"]
    link_file = os.environ["CALIPER_RUN_LINK_FILE"]
    params = json.loads(os.environ.get("CALIPER_RUN_PARAMS") or "{}")
    resume = os.environ.get("CALIPER_RUN_RESUME")
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    pace = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    exit_after_checkpoint = len(sys.argv) > 3 and sys.argv[3] == "exit-after-checkpoint"

    state = {"step": 0, "loss": 2.0, "stop": False, "saved": False}
    lr0 = 0.05
    if resume:
        with open(resume, encoding="utf-8") as f:
            saved = json.load(f)
        state["step"], state["loss"], lr0 = saved["step"], saved["loss"], saved["params"]["lr"]
        print(f"resumed from {os.path.basename(resume)} at step {state['step']}", flush=True)
    lr = caliper.param("lr", float(params.get("lr", lr0)), min=1e-6, max=0.5, log=True)

    def checkpoint() -> str:
        path = os.path.join(run_dir, f"checkpoint-{state['step']}.json")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"step": state["step"], "loss": state["loss"], "params": {"lr": lr.value}}, f)
        os.replace(path + ".tmp", path)
        state["saved"] = True
        return path

    def stop() -> str:
        state["stop"] = True
        return "stopping at the next step"

    caliper.action("caliper.checkpoint", checkpoint)
    caliper.action("caliper.stop", stop)
    server = caliper.serve()
    with open(link_file + ".tmp", "w", encoding="utf-8") as f:
        f.write(server.url)
    os.replace(link_file + ".tmp", link_file)
    print(f"serving {os.environ.get('CALIPER_RUN_ID')} with lr {lr.value} for the app "
          f"{os.environ.get('CALIPER_RUN_APP_PID')}", flush=True)

    for _ in range(steps):
        state["step"] += 1
        caliper.step(state["step"])
        if state["stop"]:
            print(f"stopped at step {state['step']}", flush=True)
            caliper.flush()
            raise SystemExit(0)
        if exit_after_checkpoint and state["saved"]:
            time.sleep(0.002)          # the server's thread sends the reply
            print(f"exits after the checkpoint at step {state['step']}", flush=True)
            os._exit(0)
        state["loss"] *= 1.0 - lr.value
        caliper.series(state["loss"], name="loss")
        print(f"step {state['step']} loss {state['loss']:.6f}", flush=True)
        time.sleep(pace)


if __name__ == "__main__":
    main()
