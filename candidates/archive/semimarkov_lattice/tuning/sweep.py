import sys, json, itertools
import devrun
if __name__ == "__main__":
    base = json.loads(sys.argv[1]); grid = json.loads(sys.argv[2]); which = sys.argv[3] if len(sys.argv) > 3 else "dfl"
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(base, **dict(zip(keys, vals)))
        o = devrun.evaluate(cfg, which, quiet=True)
        print(json.dumps(dict(zip(keys, vals))), " ".join("%s %.4f/%.4f %d/%d" % (k, *v) for k, v in o.items()), flush=True)
