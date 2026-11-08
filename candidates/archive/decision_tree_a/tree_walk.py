"""Plain-numpy inference for an exported decision tree (train.export): no scikit-learn, no torch.

Same rule as scikit-learn: go left when x <= threshold (float32 feature promoted to float64), and
for a missing (NaN) feature follow missing_left. Deterministic.
"""

import numpy as np


def leaf_index(tree, X):
    feature, threshold = tree["feature"], tree["threshold"]
    left, right, miss_left = tree["left"], tree["right"], tree["missing_left"].astype(bool)
    node = np.zeros(len(X), np.int64)
    rows = np.arange(len(X))
    active = feature[node] >= 0
    while active.any():
        r, nd = rows[active], node[active]
        x = X[r, feature[nd]].astype(np.float64)
        go_left = np.where(np.isnan(x), miss_left[nd], x <= threshold[nd])
        node[r] = np.where(go_left, left[nd], right[nd])
        active = feature[node] >= 0
    return node


def proba(tree, X):
    """(N, 3) leaf class probabilities P(H), P(A), P(D)."""
    return tree["prob"][leaf_index(tree, X)]
