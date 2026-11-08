"""Self-trained model on one game's pseudo-labels (shared by offline experiments)."""
import numpy as np

H, A, D, X = 0, 1, 2, 3


def boundary_dist(codes, blocks):
    """Frames to the nearest label change (block ends count as changes)."""
    n = len(codes)
    d = np.full(n, 0, np.int64)
    for _, a, b in blocks:
        c = codes[a:b + 1]
        m = len(c)
        ch = np.zeros(m + 1, bool)
        ch[0] = ch[m] = True
        ch[1:m] = c[1:] != c[:-1]
        pos = np.where(ch)[0]
        idx = np.arange(m)
        j = np.searchsorted(pos, idx, side="right")
        left = idx - pos[j - 1]
        right = pos[np.minimum(j, len(pos) - 1)] - idx - 1
        d[a:b + 1] = np.minimum(left, right)
    return d


def pseudo_labels(counts, K, codes, blocks, mask, min_dist, need_unan=True):
    conf = mask.copy()
    if need_unan:
        conf &= counts.max(1) >= K - 1e-9
    conf &= boundary_dist(codes, blocks) >= min_dist
    conf &= codes < 3
    return conf


def fit_predict(F, y, train_idx, perm, signs, model="tree", params=None, sym=True, mask=None):
    """Train on F[train_idx] (and the H/A-swapped copy), return symmetrised P(H, A, D) for mask rows."""
    params = dict(params or {})
    Xt = F[train_idx]
    yt = y[train_idx].astype(np.int64)
    if sym:
        Xs = Xt[:, perm] * signs
        ys = yt.copy()
        ys[yt == H] = A
        ys[yt == A] = H
        Xt = np.concatenate([Xt, Xs])
        yt = np.concatenate([yt, ys])
    rows = np.where(mask)[0] if mask is not None else np.arange(len(F))
    Fm = F[rows]
    if model == "tree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=0, **params)
        clf.fit(Xt, yt)

        def pr(Z):
            p = np.zeros((len(Z), 3))
            p[:, clf.classes_] = clf.predict_proba(Z)
            return p
    elif model == "lgb":
        import lightgbm as lgb
        rounds = params.pop("rounds", 100)
        prm = dict(objective="multiclass", num_class=3, learning_rate=0.1, num_leaves=15,
                   min_data_in_leaf=100, feature_fraction=1.0, bagging_fraction=1.0, lambda_l2=1.0,
                   num_threads=1, deterministic=True, force_row_wise=True, seed=0, verbose=-1)
        prm.update(params)
        b = lgb.train(prm, lgb.Dataset(Xt, yt, free_raw_data=True), rounds)

        def pr(Z):
            return b.predict(Z, num_threads=1)
    else:
        raise ValueError(model)
    p = pr(Fm)
    if sym:
        q = pr(Fm[:, perm] * signs)
        p = 0.5 * (p + q[:, [A, H, D]])
    P = np.full((len(F), 3), 1.0 / 3)
    P[rows] = p
    return P
