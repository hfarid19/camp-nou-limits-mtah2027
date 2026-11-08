"""Diagnostics: LOGO AUC of the spell classifier vs duration alone; per-spell counterfactual delta of
sel2 when handing back exactly one spell (is the training label aligned with the metric?)."""
import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train, data, spellctl as sc
train.load_all()
rows = []
for gi, g in enumerate(data.MATCHES):
    rest = [m for m in data.MATCHES if m != g]
    m = train.train(rest, "evtavg")
    for s in data.SEEDS:
        k = (g, s); G = train.G[k]
        p = m.predict(G["X"])
        base = G["base"]; s0 = {t: G["T"][t].f1(base)[0] for t in train.SEL2}
        for (a, b, c), q, x, ye3, ye2 in zip(G["spells"], p, G["X"], G["Y"]["evt3"], G["Y"]["evt2"]):
            cc = base.copy(); cc[a:b] = cc[a - 1]
            d = np.mean([G["T"][t].f1(cc)[0] - s0[t] for t in train.SEL2])
            rows.append((gi, x[0], q, ye3, ye2, d))
R = np.array(rows)
dur, q, y3, y2, d = R[:, 1], R[:, 2], R[:, 3], R[:, 4], R[:, 5]
print("n spells", len(R), "keep rate evt3 {:.3f} evt2 {:.3f}".format(y3.mean(), y2.mean()))
for nm, y in (("evt3", y3), ("evt2", y2), ("handback helps sel2", (d < 0).astype(float))):
    print("{:22s} AUC classifier {:.3f}  duration {:.3f}".format(nm, roc_auc_score(y, q), roc_auc_score(y, dur)))
print("sum of positive single-spell deltas (oracle hand-back, x1e3 per game-seed): {:.2f}".format(d[d > 0].sum() / 21 * 1e3))
print("share of spells where handback helps: {:.3f}; agreement of evtavg label with metric sign: {:.3f}".format(
    (d > 0).mean(), (((y3 + y2) / 2 < 0.5) == (d > 0))[d != 0].mean()))
for lo, hi in ((1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 99)):
    s = (dur >= lo) & (dur < hi)
    print("dur {}-{}s n {:4d} keep evt3 {:.2f} evt2 {:.2f}  handback helps {:.2f}  AUC(q, evt3) {:.3f}".format(
        lo, hi, s.sum(), y3[s].mean(), y2[s].mean(), (d[s] > 0).mean(),
        roc_auc_score(y3[s], q[s]) if 0 < y3[s].mean() < 1 else float("nan")))
np.save(os.path.join(train.ART, "diag_rows.npy"), R)
