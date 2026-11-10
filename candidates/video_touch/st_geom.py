"""Fit pitch(m) -> image(px) mapping and player box height per match from GSR slices.
Polynomial (deg 4) in (x,y) for u, v and box height h; robust (2 IRLS passes)."""
import numpy as np, glob, os, json, sys
D = os.path.join(os.environ['MTAH_WORK'], 'video_touch/stmeta/gsr_slices')
OUT = os.path.join(os.environ['MTAH_WORK'], 'video_touch/stmeta/geom')
os.makedirs(OUT, exist_ok=True)
def feats(x, y, deg=4):
    x = np.asarray(x) / 52.5; y = np.asarray(y) / 34.0
    return np.stack([x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)
def fit(A, b):
    w = np.ones(len(b))
    for _ in range(3):
        c = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        r = b - A @ c; s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = 1 / np.maximum(1, np.abs(r) / (2.5 * s))
    return c, np.median(np.abs(r))
res = {}
for f in sorted(glob.glob(D + '/*.npy')):
    a = np.load(f); key = os.path.basename(f)[:-4]
    ok = np.isfinite(a).all(1) & (a[:, 3] > 5)
    a = a[ok]; A = feats(a[:, 4], a[:, 5])
    cu, eu = fit(A, a[:, 1]); cv, ev = fit(A, a[:, 2]); ch, eh = fit(A, np.log(a[:, 3]))
    res[key] = dict(cu=cu.tolist(), cv=cv.tolist(), ch=ch.tolist())
    print(key, len(a), 'median abs err px u %.1f v %.1f logh %.3f' % (eu, ev, eh), 'h range', np.percentile(a[:, 3], [5, 50, 95]).round())
json.dump(res, open(OUT + '/geom.json', 'w'))
