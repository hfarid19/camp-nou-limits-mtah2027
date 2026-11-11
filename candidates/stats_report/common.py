"""Shared paths and helpers for the statistical review (candidates/stats_report)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
REPO = os.path.dirname(CAND)
WORK = os.environ["MTAH_WORK"]
os.environ.setdefault("MTAH_WORK", WORK)
STATS = os.path.join(WORK, "stats")
PROBS = os.path.join(WORK, "probs")
FIG = os.path.join(HERE, "fig")
os.makedirs(STATS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, CAND)
import bench                     # noqa: E402,F401
import possession_metrics as pm  # noqa: E402,F401
import control                   # noqa: E402,F401

MATCHES = sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "idsse", "npz")) if f.endswith(".npz"))
SEEDS = [2027, 2028, 2029]
SHORT = {m: m[-3:] for m in MATCHES}

# palette (dataviz reference instance, light mode)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK2, SURF = "#9a9994", "#0b0b0b", "#52514e", "#fcfcfb"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2})
    return plt


def runs_from_blocks(blocks):
    return [(int(a), int(b) + 1) for _, a, b in blocks]
