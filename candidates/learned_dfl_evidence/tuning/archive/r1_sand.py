import numpy as np, sys
import r1_ev as ev
sys.path.insert(0, ev.ROOT); import possession_pipeline as pp
def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
for mode in ["clean_sm", "osd_sm"]:
    ins, sur, hit, tot = [], [], 0, 0
    for r in ev.rows(mode):
        gt = r["gt"].copy()
        aux = dict(r["aux"]); aux["long_gap"] = ev.L.long_gap_gated(r["players"], r["ball"], 25, 3.0, 2.0)
        pred = ev.L.decode(r["pdd"], r["ph"], aux, r["blocks"], len(gt), 25)
        segs = pp.segments(gt)
        l = lg(r["ph"])
        for i in range(1, len(segs) - 1):
            s, e, lab = segs[i]
            if lab > 1 or not (25 <= e - s <= 75): continue
            if segs[i-1][2] != 1 - lab or segs[i+1][2] != 1 - lab: continue
            sg = 1 if lab == 0 else -1
            tot += 1
            ins.append(sg * l[s:e].mean())
            sur.append(sg * np.r_[l[max(0, s-75):s], l[e:e+75]].mean())
            hit += (pred[s:e] == lab).mean() > 0.5
    ins, sur = np.array(ins), np.array(sur)
    print(mode, "n", tot, "recovered", hit, "own-team logit inside: mean %.2f  median %.2f  share>0 %.2f | surrounding mean %.2f" % (ins.mean(), np.median(ins), (ins > 0).mean(), sur.mean()))
    print("  inside-surround diff quantiles", np.round(np.percentile(ins - sur, [10, 25, 50, 75, 90]), 2))
