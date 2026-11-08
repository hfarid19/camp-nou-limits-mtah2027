import sys, json, numpy as np, devrun
import pipeline as P
def spells(x):
    ch = np.r_[0, np.where(np.diff(x) != 0)[0] + 1, len(x)]
    return [(a, b, x[a]) for a, b in zip(ch[:-1], ch[1:]) if x[a] < 2]
def iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); return i / (max(a[1], b[1]) - min(a[0], b[0]))
if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    bins = [0, 1, 2, 4, 8, 16, 1e9]
    tp_t = np.zeros(6); n_t = np.zeros(6); tp_p = np.zeros(6); n_p = np.zeros(6)
    for gi, m in enumerate(devrun.MATCHES):
        ev, gt, blocks = devrun.load(devrun.EV + "/dfl_%s.npz" % m)
        pred, _ = P.decode_ev(ev, blocks, len(gt), 25, devrun.tab_for(gi), cfg, False)
        for _, f, l in blocks:
            g = spells(gt[f:l + 1]); p = spells(pred[f:l + 1])
            ps = np.array([(a, b) for a, b, _ in p]); 
            for a, b, lab in g:
                k = np.searchsorted(bins, (b - a) / 25, side="right") - 1
                n_t[k] += 1
                cand = [q for q in p if q[2] == lab and q[0] < b and q[1] > a]
                if any(iou((a, b), q) >= 0.5 for q in cand): tp_t[k] += 1
            for a, b, lab in p:
                k = np.searchsorted(bins, (b - a) / 25, side="right") - 1
                n_p[k] += 1
                cand = [q for q in g if q[2] == lab and q[0] < b and q[1] > a]
                if any(iou((a, b), q) >= 0.5 for q in cand): tp_p[k] += 1
    for k in range(6):
        print("dur %4.0f-%4.0f s  true %4d recall@.5 %.3f   pred %4d precision@.5 %.3f" % (bins[k], min(bins[k+1], 99), n_t[k], tp_t[k] / max(n_t[k], 1), n_p[k], tp_p[k] / max(n_p[k], 1)))
