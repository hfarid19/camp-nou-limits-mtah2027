"""Do video touches separate correct from wrong short sandwiched spells in v5c's codes?
correct = majority of truth (raw labels) over the spell equals the spell team."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_phase3 as S
runs = sys.argv[1:]
import glob
games = sorted(set(os.path.basename(f).split('_')[1] for r in runs for f in glob.glob(f'{r}/prob_*_2nd.npy')))
rows = []
for m in games:
    d = S.game_data(m, runs)
    if d is None: continue
    c, g = d['codes'], d['gt']
    for _, a, b in d['blocks']:
        sp = S.spells(c, a, b)
        for j in range(1, len(sp) - 1):
            s0, s1, team = sp[j]
            if team not in (S.pm.H, S.pm.A) or s1 - s0 + 1 > 75: continue
            if not (sp[j-1][2] == sp[j+1][2] != team and sp[j-1][2] in (S.pm.H, S.pm.A)): continue
            vals, cnt = np.unique(g[s0:s1+1], return_counts=True)
            correct = vals[np.argmax(cnt)] == team
            q = d['p'][max(0, s0 - 3):s1 + 1]
            rows.append((m, s1 - s0 + 1, correct, q.max()))
    # team-specific evidence
R = np.array([(r[1], r[2], r[3]) for r in rows], float)
print('candidate spells', len(R), 'share correct %.3f' % R[:, 1].mean())
from itertools import product
for thr in (0.2, 0.3, 0.4, 0.5):
    has = R[:, 2] >= thr
    print(f'max p >= {thr}: n {has.sum():4d} correct {R[has,1].mean():.3f} | without: n {(~has).sum():4d} correct {R[~has,1].mean() if (~has).any() else float("nan"):.3f}')
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print('AUC(max touch prob in spell -> correct) %.3f; AUC(duration) %.3f' % (auc(R[:, 2], R[:, 1]), auc(R[:, 0], R[:, 1])))
