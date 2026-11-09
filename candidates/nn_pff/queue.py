#!/usr/bin/env python3
"""Tiny job queue: runs the lines of a jobs file (shell commands) in order, keeping at most --max
heavy processes on the machine (counted with ps: train_*, prep_data, feat_data, train_lgb, bench,
run_pff; including processes not started by this queue; --pat counts only matching ones, e.g. GPU jobs).
Lines appended later are picked up.
State: <jobs>.state (one started line index per line). Each job's output -> <logdir>/job_<i>.log.

    python candidates/nn_pff/queue.py JOBS --max 6 --logdir DIR
"""
import argparse
import os
import subprocess
import time

PAT = ("train_mstcn.py", "train_tcn.py", "train_radar.py", "prep_data.py", "feat_data.py", "train_lgb.py",
       "bench.py", "run_pff.py", "bench_pff.py run")


def heavy(pat=None):
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout.splitlines()
    out = [l for l in out if l.split() and "python" in l.split()[0].lower() and "queue.py" not in l]  # no shells / queues
    if pat:
        return sum(1 for l in out if "python" in l.lower() and pat in l)
    return sum(1 for l in out if "python" in l.lower() and any(p in l for p in PAT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--pat", default=None, help="count only processes whose command contains this "
                    "(e.g. '--device mps': GPU slots)")
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    st = a.jobs + ".state"
    while True:
        lines = [l.rstrip("\n") for l in open(a.jobs)]
        done = {int(x) for x in open(st).read().split()} if os.path.exists(st) else set()
        todo = [i for i, l in enumerate(lines) if l.strip() and not l.startswith("#") and i not in done]
        if todo and heavy(a.pat) < a.max:
            i = todo[0]
            with open(st, "a") as h:
                h.write("{}\n".format(i))
            log = open(os.path.join(a.logdir, "job_{}.log".format(i)), "w")
            log.write("# " + lines[i] + "\n")
            log.flush()
            subprocess.Popen(["/bin/zsh", "-c", lines[i]], stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
            time.sleep(20)
            continue
        time.sleep(30)


if __name__ == "__main__":
    main()
