"""Phase 3 on SoccerTrack: do video touch peaks improve ensemble_ctrl_v5c's codes?

For each held-out game with touch probs for both halves (runs/<run>/prob_<m>_<h>.npy) and saved v5c
codes ($MTAH_WORK/probs/soccertrack/ensemble_ctrl_v5c/st2_<m>_2027_codes.npy):
  peaks   = NMS peaks of p (radius 6) mapped to bench rows (npz half / 1-based frame)
  team    = team of the nearest degraded-tracking player to the degraded ball (gaps interpolated) at the peak
Rules (pm codes: H=0? -- read from possession_metrics), applied inside each block:
  snap(thr, W): each H<->A switch moves to the nearest peak within +-W rows attributed to the new team
  insert(thr, L): inside a team-a spell, a peak of team b != a starts a b-spell lasting to the next
                  peak attributed to a (only if that is <= L rows away; else nothing)
  remove(thr, L): a team-b spell shorter than L between two team-a spells, containing no b-peak >= thr,
                  is handed back to a
usage: st_phase3.py RUN [RUN2 ...]   (prob files are pooled from all given run dirs)
"""
import sys, os, glob, itertools, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), f'{REPO}/candidates/soccertrack', f'{REPO}/candidates', REPO, f'{REPO}/evaluation']
if 'MTAH_WORK' not in os.environ: sys.exit('set MTAH_WORK to the work directory')
import bench_st as B
import possession_metrics as pm
from vt_eval import nms_peaks

W = os.environ['MTAH_WORK']
CODES = f'{W}/probs/soccertrack/ensemble_ctrl_v5c'


def interp(a):
    a = a.copy(); i = np.arange(len(a))
    for k in range(a.shape[1]):
        ok = np.isfinite(a[:, k])
        if ok.sum() > 1:
            a[~ok, k] = np.interp(i[~ok], i[ok], a[ok, k])
    return a


def game_data(m, runs):
    z = np.load(f'{W}/soccertrack/npz/ST-{m}.npz')
    half, frame = z['half'], z['frame']
    p = np.zeros(len(half), np.float32)
    for h, hn in ((1, '1st'), (2, '2nd')):
        f = [f'{r}/prob_{m}_{hn}.npy' for r in runs if os.path.exists(f'{r}/prob_{m}_{hn}.npy')]
        if not f:
            return None
        q = np.load(f[0]); idx = np.where(half == h)[0]
        p[idx] = q[np.clip(frame[idx] - 1, 0, len(q) - 1)]
    slots, slot_team, ball, gt, blocks, sign = B.st2_prepared(m)
    codes = np.load(f'{CODES}/st2_{m}_2027_codes.npy')
    bl = interp(ball.astype(float))
    return dict(m=m, p=p, slots=slots, slot_team=slot_team, ball=bl, gt=gt, blocks=blocks, codes=codes, half=half)


def peaks_with_team(d, thr):
    pk = []
    for _, a, b in d['blocks']:
        q = np.full_like(d['p'], -1); q[a:b + 1] = d['p'][a:b + 1]
        pk += list(nms_peaks(q, thr, 6))
    pk = np.array(sorted(pk), int)
    team = np.full(len(pk), -1)
    for i, r in enumerate(pk):
        dist = np.linalg.norm(d['slots'][r] - d['ball'][r], axis=1)
        if os.environ.get('ORACLE_TEAM'):
            g = d['gt'][min(r + 2, len(d['gt']) - 1)]
            team[i] = {pm.H: 0, pm.A: 1}.get(g, -1)
        elif np.isfinite(dist).any():
            team[i] = d['slot_team'][np.nanargmin(dist)]
    return pk, team


TEAMCODE = {0: pm.H, 1: pm.A}


def spells(c, a, b):
    out = []; s = a
    for i in range(a + 1, b + 2):
        if i == b + 1 or c[i] != c[s]:
            out.append((s, i - 1, c[s])); s = i
    return out


def apply(d, rule, thr, L):
    c = d['codes'].copy()
    pk, tm = peaks_with_team(d, thr)
    tc = np.array([TEAMCODE.get(t, -9) for t in tm])
    alive = (pm.H, pm.A)
    for _, a, b in d['blocks']:
        sel = (pk >= a) & (pk <= b); P, T = pk[sel], tc[sel]
        if rule == 'snap':
            sw = [i for i in range(a + 1, b + 1) if c[i] in alive and c[i - 1] in alive and c[i] != c[i - 1]]
            for s in sw:
                new = c[s]; old = c[s - 1]
                cand = P[(T == new) & (np.abs(P - s) <= L)]
                if len(cand):
                    t = cand[np.argmin(np.abs(cand - s))]
                    if t < s:
                        seg = slice(t, s); ok = np.all(c[seg] == old)
                        if ok: c[seg] = new
                    elif t > s:
                        seg = slice(s, t); ok = np.all(c[seg] == new)
                        if ok: c[seg] = old
        elif rule == 'insert':
            for k in range(len(P)):
                t, team = P[k], T[k]
                if team not in alive or c[t] not in alive or c[t] == team:
                    continue
                host = c[t]
                nxt = [P[j] for j in range(k + 1, len(P)) if T[j] == host]
                if not nxt or nxt[0] - t > L:
                    continue
                seg = slice(t, nxt[0])
                if np.all(np.isin(c[seg], alive)):
                    c[seg] = team
        elif rule == 'remove':
            sp = spells(c, a, b)
            for j in range(1, len(sp) - 1):
                s0, s1, team = sp[j]
                if team not in alive or s1 - s0 + 1 > L:
                    continue
                if sp[j - 1][2] == sp[j + 1][2] and sp[j - 1][2] in alive and sp[j - 1][2] != team:
                    if not np.any((P >= s0 - 3) & (P <= s1) & (T == team)):
                        c[s0:s1 + 1] = sp[j - 1][2]
    return c


def team_accuracy(d, thr=0.4):
    """Attributed team vs the label 2 frames after each peak matched (+-5) to a truth team-onset."""
    pk, tm = peaks_with_team(d, thr)
    g = d['gt']; ok = n = 0
    for r, t in zip(pk, tm):
        r2 = min(r + 2, len(g) - 1)
        if g[r2] in (pm.H, pm.A) and t >= 0:
            n += 1; ok += TEAMCODE[t] == g[r2]
    return ok / max(n, 1), n


def main():
    runs = sys.argv[1:]
    games = sorted(set(os.path.basename(f).split('_')[1] for r in runs for f in glob.glob(f'{r}/prob_*_2nd.npy')))
    data = [game_data(m, runs) for m in games]
    data = [d for d in data if d is not None and os.path.exists(f"{CODES}/st2_{d['m']}_2027_codes.npy")]
    print('games', [d['m'] for d in data])
    for d in data:
        print(d['m'], 'team attribution accuracy (peaks thr .4 vs label +2f): %.3f n %d' % team_accuracy(d))
    base = {d['m']: B.score_codes(d['codes'], d['m']) for d in data}
    keys = ['raw', 'fine3', 'flag', 'T1', 'macroF1']
    print('v5c base', {k: round(np.mean([base[g][k] for g in base if k in base[g]]), 4) for k in keys})
    grid = {'snap': [(t, L) for t in (0.3, 0.4, 0.5) for L in (6, 12, 25)],
            'insert': [(t, L) for t in (0.4, 0.5, 0.6) for L in (25, 50, 75)],
            'remove': [(t, L) for t in (0.2, 0.3, 0.4) for L in (25, 50, 75)],
            'removeall': [(9.9, L) for L in (50, 75)]}   # control: remove every candidate, no video
    res = {}
    for rule, g in grid.items():
        for thr, L in g:
            sc = {d['m']: B.score_codes(apply(d, rule.replace('removeall', 'remove'), thr, L), d['m']) for d in data}
            diff = {k: [sc[m][k] - base[m][k] if (k in sc[m] and k in base[m]) else np.nan for m in sc] for k in keys}
            res[(rule, thr, L)] = diff
            print(f'{rule:6s} thr {thr} L {L}: ' + ' '.join(f'{k} {np.nanmean(v):+.4f}' for k, v in diff.items()),
                  ' per-game fine3', np.round(diff['fine3'], 4).tolist(), flush=True)
    # nested (leave-one-game-out) choice on fine3 within each rule
    if len(data) > 1:
        ms = [d['m'] for d in data]
        for rule in grid:
            out = []
            for i, m in enumerate(ms):
                best = max(grid[rule], key=lambda tl: np.mean([res[(rule,) + tl]['fine3'][j] for j in range(len(ms)) if j != i]))
                out.append(res[(rule,) + best]['fine3'][i])
            print(f'nested {rule}: fine3 {np.mean(out):+.4f} per game {np.round(out, 4).tolist()}')
            outf = []
            for i, m in enumerate(ms):
                best = max(grid[rule], key=lambda tl: np.nanmean([res[(rule,) + tl]['flag'][j] for j in range(len(ms)) if j != i]))
                outf.append(res[(rule,) + best]['flag'][i])
            print(f'nested {rule} (chosen+scored on flag): flag {np.nanmean(outf):+.4f} per game {np.round(outf, 4).tolist()}')


if __name__ == '__main__':
    main()
