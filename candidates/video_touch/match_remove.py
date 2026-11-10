"""Apply the 'remove' rule to ensemble_ctrl_v5c's match output (no labels used).

A team-b spell of <= L frames between two team-a spells is handed back to a when the video detector
shows no touch peak (>= thr) attributed to b in [start-3, end]. Touch probs: CPU inference with fixed
threads on the crops around each candidate spell (mean of the given runs' models). Team: nearest
tracked player (match tracking, slot team from the ATD metadata) to the tracking ball (gaps interpolated).
usage: match_remove.py --runs runs/foldA,runs/foldB --thr 0.4 --L 50 --threads 3 [--out file.csv]
"""
import sys, os, csv, time, hashlib, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)
from vt_model import TouchNet
from vt_eval import nms_peaks
from train_touch import predict_half, CROPS
import possession_pipeline as pp

ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True); ap.add_argument('--thr', type=float, required=True)
ap.add_argument('--L', type=int, required=True); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--out', default=None); ap.add_argument('--margin', type=int, default=48)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.use_deterministic_algorithms(True)

rows = list(csv.reader(open(f'{REPO}/candidates/ensemble_ctrl_v5c/submission.csv')))
hdr, rows = rows[0], rows[1:]
lab = np.array([r[-1] for r in rows])
sp = []; s = 0
for i in range(1, len(lab) + 1):
    if i == len(lab) or lab[i] != lab[s]:
        sp.append((s, i - 1, lab[s])); s = i
cand = [j for j in range(1, len(sp) - 1) if sp[j][2] in 'HA' and sp[j][1] - sp[j][0] + 1 <= a.L
        and sp[j - 1][2] == sp[j + 1][2] and sp[j - 1][2] in 'HA' and sp[j - 1][2] != sp[j][2]]
print('candidate spells', len(cand))

models = []
for r in a.runs.split(','):
    m = TouchNet(T=64, pretrained=False); m.load_state_dict(torch.load(f'{r}/model.pt', map_location='cpu')); models.append(m.eval())
arr = np.load(f'{CROPS}/crops_BARATH_full.npy', mmap_mode='r')
z = np.load(os.path.join(os.environ['MTAH_WORK'], 'match_cache.npz')); pl, b = z['pl'], z['b'].astype(float)
i = np.arange(len(b))
for k in range(2):
    ok = np.isfinite(b[:, k]); b[~ok, k] = np.interp(i[~ok], i[ok], b[ok, k])
slot_team = pp.read_meta(f'{REPO}/data/tactical/BAR-ATH_tactical_atd.xml')[0]

out = lab.copy(); t0 = time.time(); nfr = 0; changed = []
for j in cand:
    s0, s1, team = sp[j]
    w0, w1 = max(0, s0 - a.margin), min(len(arr), s1 + 1 + a.margin)
    if w1 - w0 < 64:
        w1 = min(len(arr), w0 + 64)
    p = np.mean([predict_half(m, arr[w0:w1], 64, 128, 'cpu') for m in models], 0); nfr += (w1 - w0) * len(models)
    pk = nms_peaks(p, a.thr, 6) + w0
    pk = pk[(pk >= s0 - 3) & (pk <= s1)]
    tms = []
    for r in pk:
        d = np.linalg.norm(pl[r] - b[r], axis=1)
        if np.isfinite(d).any():
            tms.append('H' if slot_team[np.nanargmin(d)] == 0 else 'A')
    if team not in tms:
        out[s0:s1 + 1] = sp[j - 1][2]; changed.append((s0, s1 - s0 + 1))
dt = time.time() - t0
print(f'removed {len(changed)} of {len(cand)} spells, {sum(c[1] for c in changed)} frames; '
      f'inference {nfr} model-frames in {dt:.0f}s ({nfr / max(dt, 1e-9):.0f} fps, {a.threads} threads)')
if a.out:
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(hdr)
        for r, l in zip(rows, out):
            w.writerow(r[:-1] + [l])
    print('sha256', hashlib.sha256(open(a.out, 'rb').read()).hexdigest())
