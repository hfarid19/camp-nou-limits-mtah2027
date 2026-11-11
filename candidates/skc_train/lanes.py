#!/usr/bin/env python3
"""Tiny local job runner for the round-5 "+SKC" scoring on the Mac (at most --lanes jobs at once).

    python -I candidates/skc_train/lanes.py JOBS --lanes 2 --logdir DIR

JOBS: one job per line, "REQ <file>[,<file>...] :: <shell command>" or just "<shell command>"; '#' lines ignored.
The first not-yet-started line whose REQ files all exist is started next (so later lines can run while earlier
ones wait for models). Lines appended later are picked up. State: JOBS.state (started line indices). Exits when every
line has started and finished and the file "JOBS.stop" exists (touch it to let the runner end).
"""
import argparse
import os
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--logdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    st = a.jobs + ".state"
    running = {}
    while True:
        for i, p in list(running.items()):
            if p.poll() is not None:
                del running[i]
        lines = [l.rstrip("\n") for l in open(a.jobs)]
        done = {int(x) for x in open(st).read().split()} if os.path.exists(st) else set()
        todo = [i for i, l in enumerate(lines) if l.strip() and not l.startswith("#") and i not in done]
        if not todo and not running and os.path.exists(a.jobs + ".stop"):
            return
        if len(running) < a.lanes:
            for i in todo:
                l = lines[i]
                req, cmd = (l[4:].split(" :: ", 1) if l.startswith("REQ ") else ("", l))
                if not all(os.path.exists(f) for f in req.split(",") if f.strip()):
                    continue
                with open(st, "a") as h:
                    h.write("{}\n".format(i))
                log = open(os.path.join(a.logdir, "job_{}.log".format(i)), "w")
                log.write("# " + cmd + "\n")
                log.flush()
                running[i] = subprocess.Popen(["/bin/zsh", "-c", cmd], stdout=log, stderr=subprocess.STDOUT)
                break
        time.sleep(10)


if __name__ == "__main__":
    main()
