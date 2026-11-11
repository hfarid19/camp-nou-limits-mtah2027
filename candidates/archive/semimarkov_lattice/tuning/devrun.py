"""Dev harness: cache LDE evidence (fold models on DFL) and score decoder configs quickly.
   python devrun.py cache            # builds ev caches
   python devrun.py eval '{json cfg}' [dfl|dev|both]
"""
import os, sys, json, time, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "2")
REPO = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "..", ".."))
CAND = REPO + "/candidates/semimarkov_lattice"
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")
S = WORK
A = S + "/arch/semimarkov_lattice"
EV = A + "/ev"
sys.path.insert(0, REPO + "/candidates"); sys.path.insert(0, CAND)
import numpy as np
import bench
import possession_metrics as pm
import tune_on_sample_data as tsd

MATCHES = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))
DEVG = [(g, s) for g in bench.DEV for s in bench.SEEDS]


def cache():
    import lightgbm as lgb
    os.makedirs(EV, exist_ok=True)
    folds = json.load(open(A + "/folds.json"))
    real = lgb.Booster
    import pipeline as P
    for gi, m in enumerate(MATCHES):
        out = EV + "/dfl_%s.npz" % m
        if os.path.exists(out):
            continue
        def booster(model_file=None):
            sha = hashlib.sha256(open(model_file, "rb").read()).hexdigest()
            key = next(k for k in folds if sha.startswith(k))
            folder, kind = folds[key]
            full = real(model_file=model_file)
            fold = real(model_file=os.path.join(folder, "fold_%s_%d.txt" % (kind, gi)))
            assert fold.feature_name() == full.feature_name()
            return bench._FoldBooster(fold, full.num_trees())
        lgb.Booster = booster
        P.L._MODELS.clear()
        slots, st, ball, gt, blocks, sign = bench.dfl_prepared(m, 2027)
        t = time.time()
        ev = P.evidence(slots, st, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
        np.savez(out, gt=gt, blocks=np.array(blocks), **ev)
        print(m, round(time.time() - t, 1), flush=True)
    lgb.Booster = real
    P.L._MODELS.clear()
    for g, s in DEVG:
        out = EV + "/dev_%s_%d.npz" % (g, s)
        if os.path.exists(out):
            continue
        slots, st, ball, gt, blocks, sign = bench.prepared(g, s)
        ev = P.evidence(slots, st, ball, sign, blocks, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
        np.savez(out, gt=gt, blocks=np.array(blocks), **ev)
        print(g, s, flush=True)


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    blocks = [tuple(int(v) for v in b) for b in d.pop("blocks")]
    gt = d.pop("gt")
    return d, gt, blocks


def tab_for(gi):
    p = CAND + "/durations.npz" if gi is None else A + "/folds/fold_dur_%d.npz" % gi
    z = np.load(p)
    return {k: z[k] for k in z.files}


def run_one(job):
    kind, name, gi, cfg, want_post = job
    import pipeline as P
    ev, gt, blocks = load(EV + "/%s_%s.npz" % (kind, name))
    pred, proba = P.decode_ev(ev, blocks, len(gt), 25, tab_for(gi), cfg, want_post)
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def evaluate(cfg, which="dfl", want_post=False, procs=2, quiet=False):
    jobs = []
    if which in ("dfl", "both"):
        jobs += [("dfl", m, gi, cfg, want_post) for gi, m in enumerate(MATCHES)]
    if which in ("dev", "both"):
        jobs += [("dev", "%s_%d" % gs, None, cfg, want_post) for gs in DEVG]
    from multiprocessing import Pool
    with Pool(procs) as p:
        res = p.map(run_one, jobs)
    out = {}
    for kind in ("dfl", "dev"):
        rr = [r for j, r in zip(jobs, res) if j[0] == kind]
        if rr:
            out[kind] = (np.mean([r[1] for r in rr]), np.mean([r[2] for r in rr]),
                         sum(r[3] for r in rr), sum(r[4] for r in rr))
            if not quiet:
                for r in rr:
                    print("   %-22s %.4f %.4f %d/%d" % r)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cache()
    else:
        cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        which = sys.argv[3] if len(sys.argv) > 3 else "dfl"
        t = time.time()
        o = evaluate(cfg, which, want_post=len(sys.argv) > 4)
        for k, v in o.items():
            print("%s %s spellF1 %.4f macroF1 %.4f poss %d/%d  (%.0fs)" % (k, json.dumps(cfg), v[0], v[1], v[2], v[3], time.time() - t))
