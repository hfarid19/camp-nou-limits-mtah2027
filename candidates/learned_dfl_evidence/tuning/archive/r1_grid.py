import sys, json, multiprocessing as mp
import r1_ev as ev
import os
MODES = os.environ.get("MODES", "clean,osd").split(",")
def one(cfg):
    return cfg, {m: ev.score(m, cfg)[:3] for m in MODES}
if __name__ == "__main__":
    cfgs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    for m in MODES: ev.rows(m)
    with mp.get_context("fork").Pool(8) as p:
        for cfg, r in p.imap(one, cfgs):
            print(json.dumps(cfg), *[r[m] for m in MODES], "mean %.4f" % (sum(r[m][0] for m in MODES) / len(MODES)), flush=True)
