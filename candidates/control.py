"""The challenge's "controlled possession" rule as an operation on a label sequence.

CHALLENGE.md: a change of possession needs the opponent to gain CONTROLLED possession. A touch
without control, a deflection that does not intercept, or an aerial duel won without later control
does not count, and loose balls stay with the last team; one suggested proxy is "a player keeps
the ball for a few seconds". fold() applies that rule. Every possession spell shorter than T
seconds that directly follows the other team's possession is given back to that team, repeated
until nothing changes. Spells that follow a stoppage are kept: after a restart there is always a
clear possessor.

The bench applies it to DFL labels (the DFL flag flips on every touch) to score against the
competition's meaning, and candidates may apply it to their own output.
"""

import possession_metrics as pm


def fold(codes, T, runs, fps=25):
    """Copy of codes with possession spells shorter than T s handed back to the preceding team.
    runs: [(start, stop)] half-open graded spans (possession_metrics.valid_runs)."""
    if not T:
        return codes
    out = codes.copy()
    for _ in range(10):
        changed = False
        for a, b, c in pm.segments(out, runs):
            if c in (pm.H, pm.A) and b - a < T * fps and a > 0 and out[a - 1] in (pm.H, pm.A) \
                    and out[a - 1] != c:
                out[a:b] = out[a - 1]
                changed = True
        if not changed:
            break
    return out
