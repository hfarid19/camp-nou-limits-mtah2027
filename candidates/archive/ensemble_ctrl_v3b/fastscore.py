"""spell_f1_mean of possession_metrics.evaluate, computed fast (same matching, sorted-interval overlaps).
Checked equal to pm.evaluate(...)["spell_f1_mean"] in train.py --check."""
import numpy as np
import possession_metrics as pm


class Target(object):
    def __init__(self, gt):
        self.runs = pm.valid_runs(gt)
        sp = pm.possession_spells(gt, self.runs)
        self.s = np.array([x[0] for x in sp]); self.e = np.array([x[1] for x in sp])
        self.c = np.array([x[2] for x in sp])

    def f1(self, pred):
        filled = pm.fill_abstentions(pred, self.runs)
        sp = pm.possession_spells(filled, self.runs)
        ps = np.array([x[0] for x in sp]); pe = np.array([x[1] for x in sp]); pc = np.array([x[2] for x in sp])
        ov = []
        # pred spells j overlapping gt spell i: ps[j] < e[i] and pe[j] > s[i]
        lo = np.searchsorted(pe, self.s, side="right")
        hi = np.searchsorted(ps, self.e, side="left")
        for i in range(len(self.s)):
            for j in range(lo[i], hi[i]):
                if pc[j] != self.c[i]:
                    continue
                inter = min(self.e[i], pe[j]) - max(self.s[i], ps[j])
                if inter <= 0:
                    continue
                union = max(self.e[i], pe[j]) - min(self.s[i], ps[j])
                ov.append((inter / float(union), i, j))
        ov.sort(key=lambda t: (-t[0], t[1], t[2]))
        ng, npd = len(self.s), len(ps)
        f1s = []
        for thr in pm.SPELL_IOUS:
            tp = int(pm.matched_gt_flags(ov, ng, thr).sum())
            p = tp / float(npd) if npd else 0.0
            r = tp / float(ng) if ng else 0.0
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        return float(np.mean(f1s)), npd
