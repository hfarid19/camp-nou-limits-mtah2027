# ensemble_ctrl_v5a: boundary precision and short possessions (built, measured, NOT recommended)

`pipeline.py` is `ensemble_ctrl_v4` (vote of the raw-flag TCN members + ball_radar, merge 1 s,
control fold 1.25 s, prekick_dead) plus one step, `insert_short`: inside every predicted H/A spell,
a run of at least 0.6 s where the mean of the two TCN members' P(opponent) exceeds 0.8 becomes an
opponent spell. It is the best configuration of the best of four ideas tried in this area, and on
the bench it is indistinguishable from v4. **Ship v4, not this.** The value of this folder is the
measurement: on our bench the short-possession and boundary headroom is not reachable from the
signals we have.

## Why this area

The organisers' shift fingerprint (EVALUATION.md) says the truth keeps many short spells (about
35% under 3 s). On bench v2, v4 recalls true spells at IoU 0.5 like this (raw DFL flag / 1 s fold):

| true spell length | < 1 s | 1-2 s | 2-3 s | 3-8 s | > 8 s |
|---|---|---|---|---|---|
| raw flag, recall | 0.01 (n 1152) | 0.31 (1605) | 0.51 (1197) | 0.72 (2769) | 0.92 (2460) |
| 1 s fold, recall | 0.22 (45) | 0.34 (1329) | 0.56 (1050) | 0.74 (2709) | 0.92 (2538) |

The 1-3 s spells are the headroom. Boundary jitter is not: see idea 1.

## Ideas, in the order they were ruled out (`candidates/_scratch/v5a_*.py`)

All numbers are on bench v2 (7 DFL games x 3 degradation seeds, leave-one-game-out codes), paired
against v4 on the same game and seed, SE over the 7 game means. "Nested" means the rule's
parameters were chosen on 6 games and scored on the 7th.

### 1. Boundary snapping to touch peaks: oracle says no (`v5a_diag.py`)

Before building a detector, the upper bound: move every predicted switch to the **nearest true
switch** within a window.

| oracle snap of all H/A/D switches | +-5 frames | +-12 frames (0.5 s) | +-25 frames (1 s) |
|---|---|---|---|
| fine4 gain, perfect snapping | +0.0028 | +0.0050 | +0.0108 |

A real detector gets a fraction of that. The reason: v4's boundaries are already centred. For
matched spells (IoU >= 0.3), the H<->A start offset vs the raw flag has median -0.5 frames, IQR 10;
D->alive starts are exact (89% within +-1 frame). Alive->D ends are 8 frames early (median), but
those spells are long and snapping them perfectly within 1 s gives +0.002. A uniform shift of all
H<->A switches by -6...+4 frames never gains (best +0.0000 at +1). Not built.

### 2a. Keep the short spells the 1.25 s fold removes: too few, no signal (`v5a_short.py`)

The fold removes only 144 spells over the 21 game-seeds (the vote's 1 s merge already removed
everything shorter). Of those, 56% have the right team, 33% match a true raw-flag spell at IoU
0.5, and no feature separates them (unanimous members: 9 cases; touch inside: 68% right team but
36% IoU match). Of 1341 true 1-3 s spells v4 misses (1 s fold labels), only 402 exist in **any**
member's output at IoU 0.5, 49 in the smoothed vote, 10 in all three members. Recovery has to come
from below the members' hard labels.

### 2b. Soft insertion from the members' probabilities: this folder (`v5a_soft.py`)

Grid: pooled source (mean of 3 members, mean of the 2 TCNs, TCN alone, MS-TCN alone) x threshold
0.5-0.8 x minimum run 0.2-0.8 s x smoothing 0/3 frames.

Discriminability: inside the missed true 1-3 s spells the smoothed max of pooled P(new team) has
median 0.28 (q75 0.46); it exceeds 0.5 in 21% of them, against 0.7% of 1.5 s control windows inside
correct spells. The members' per-frame probabilities simply do not see these spells.

| nested, vs v4 | fine4 | sel2 | raw | T0.5 | T1 | evt1 |
|---|---|---|---|---|---|---|
| soft insertion | -0.0014 +- 0.0015 (3/7 games, 6/21 game-seeds) | -0.0019 +- 0.0011 | -0.0007 +- 0.0016 | -0.0008 | -0.0026 | -0.0016 |

Nested picks: (mean2, 0.8, 0.6 s) x3, (mean2, 0.8, 0.4 s) x2, (mean3, 0.8, 0.6 s), (mean3, 0.6,
0.8 s). The majority pick is what `pipeline.py` fixes; in-sample it is +0.0007 fine4. It changes
0-91 frames per game-seed and adds 0-7 spells.

### 2c. MS-TCN boundary-peak pairs (ASRF flip at ensemble level) (`v5a_pb_touch.py`)

The hand-over branch of `handover_mstcn_v2` was extracted under the same LOGO fold models
(`v5a_extract_pb.py`; class probs reproduce the saved ones to 4 decimals). It does see the
missed spells: 70% have an H<->A peak > 0.5 within +-8 frames of their start, 43% at both ends,
against 13% of control windows. But it fires on much else: of 8040 adjacent-peak pairs (thr 0.3,
0.3-3 s apart) inside v4 spells, only 20% bracket an opponent spell in the truth; 23% at peak
height >= 0.5; 42% when the pooled class mass also leans >= 0.3 to the opponent; 56% at mass
>= 0.5 (n 212, about 10 per game-seed). Inserting a false spell also splits its host, so the
break-even precision is around 60%.

| nested, vs v4 | fine4 | sel2 | raw |
|---|---|---|---|
| boundary pairs (pick thr 0.5, mass 0.5, <= 1.25 s, 7/7 folds) | -0.0060 +- 0.0021 (1/7) | -0.0155 +- 0.0023 | -0.0019 +- 0.0019 |

### 2d. Touch pairs (`v5a_pb_touch.py`)

Touch detector from `_scratch/step0.py` (SG-smoothed ball acceleration > 5 m/s2 or direction
change > 10 deg, nearest player within 1.5 m). Rule: >= n opponent touches with no touch by the
possessing team between them, spanning 0.2-1.25 s (or 3 s), become an opponent spell to the last
touch + hold; optional pooled-mass gate. Team attribution is the problem: inside the missed true
spells of team c there are on average 3.3 touches by c but 5.1 by the other team (84% of such
spells contain one). Only 10% of missed spells show two clean new-team touches, against 1.6% of
control windows.

| nested, vs v4 | fine4 | sel2 | raw |
|---|---|---|---|
| touch pairs (pick n 3, >= 0.5 s, hold 0.5 s, mass 0.2) | -0.0057 +- 0.0020 (1/7) | -0.0127 +- 0.0021 | -0.0022 +- 0.0023 |

Choosing jointly over 2b-2d nested gives -0.0088 +- 0.0018 (0/7).

### Metrica dev cross-check (`v5a_metrica.py`)

Sample Games 1+2, practice labels, 2 seeds; v4 rebuilt offline from the dev member codes
(0.7083; live 0.7084).

| | spellF1 | vs v4 | wins |
|---|---|---|---|
| soft insertion (this folder) | 0.7086 | +0.0003 | 1/4 |
| boundary pairs | 0.7044 | -0.0040 | 0/4 |
| touch pairs | 0.6900 | -0.0184 | 0/4 |

### 3. Restart / stoppage boundary refinement from player speed or the kick (`v5a_restart.py`)

Suggested from other teams' repos: move D->alive to the kick (the ball reappears already moving;
extrapolate back to the taker) and alive->D to the fast-to-slow change in player speed. The oracle
above bounds it on the bench: perfect D->alive snapping within +-1 s gains +0.0003 fine4,
alive->D +0.0019, because every member already snaps D ends to the ends of long ball gaps and the
bench's D->alive starts are exact (89% within +-1 frame). The label-free check of whether the
match differs: at the 131 reappearances after gaps > 3 s on BAR-ATH the ball is indeed already
moving (median 12.3 m/s) but it is 0.7 m from the nearest player, so the back-extrapolated lag to
the kick is median 2 frames, q75 5, q90 12; 5% exceed 1 s. On the bench the same statistic is
median 2, q90 9 frames, and the reappearance coincides with the true D->alive switch (q10/50/90
all 0 frames). A 2-frame correction is worth nothing (a 1-frame offset is free in spellF1), and
the tail is too small to move a game score. Not built; could not be validated on DFL anyway.

### What cannot be validated

The independent reviewer measured boundary jitter IQR about 12 frames on the match against 3 on
the bench, and the round-2 notes trace the match's extra member disagreement to dead-vs-live
flickers while the ball is outside the lines. Any fix that acts only on the match (Smart Tagging
events, match-specific flicker rules) has no DFL or Metrica counterpart and cannot be validated
without match labels, which we may not use.

## Live results

| | fine4 | sel2 | ctrl_mean | raw spellF1 | macroF1 | Metrica dev (offline) | match possessions |
|---|---|---|---|---|---|---|---|
| ensemble_ctrl_v5a | 0.7103 | 0.7399 | 0.7183 | 0.6751 | 0.9274 | 0.7086 | 324 |
| ensemble_ctrl_v4 | 0.7095 | 0.7409 | 0.7193 | 0.6737 | 0.9272 | 0.7084 | 322 |

- **DFL bench v2, live:** `bench.py --dfl2` reproduces the offline codes on all 21 game-seeds
  (0 differing frames). Paired vs v4: fine4 **+0.0007 +- 0.0005 (10/21)** in-sample, sel2 -0.0011
  +- 0.0009 (3/21). The honest number is the nested one above, -0.0014 +- 0.0015: the parameters
  were chosen on these games. Possessions over the 21 game-seeds: 6720 vs 6691 for v4 (truth: 9183
  raw-flag spells), so the rule adds about 1.4 spells per game-seed.
- **Match:** `submission.csv` is valid (`validate_submission.py`), 199,319 rows, 324 possessions of
  mean 10.3 s, dead share 0.476, no X inside the halves. Two runs are `cmp`-identical (sha256 `008aee2a…`; v4's is `2ea055d4…`). 324
  possessions against v4's 322: the rule inserts two spells on the match.
- **Verdict:** every idea in this area is bounded (boundaries) or below break-even precision
  (short spells) on the bench; the one rule kept is noise. Do not ship over v4.

## Reproduce

```
OMP_NUM_THREADS=2 python candidates/bench.py candidates/ensemble_ctrl_v5a/pipeline.py --dfl2 \
    --folds $MTAH_WORK/arch/ensemble_ctrl_v2/folds.json --save-probs $MTAH_WORK/probs/ensemble_ctrl_v5a
python candidates/score_sel2.py $MTAH_WORK/probs/ensemble_ctrl_v5a --ref $MTAH_WORK/probs/ensemble_ctrl_v4
python candidates/bench.py candidates/ensemble_ctrl_v5a/pipeline.py --match
python evaluation/validate_submission.py candidates/ensemble_ctrl_v5a/submission.csv
```

Offline harness: `candidates/_scratch/v5a_diag.py`, `v5a_short.py`, `v5a_soft.py`,
`v5a_extract_pb.py`, `v5a_pb_touch.py`, `v5a_metrica.py`. Extracted boundary probabilities and
touch arrays: `$MTAH_WORK/probs/ensemble_ctrl_v5a_pb/`. No new models were trained; inference is
numpy/scipy/lightgbm only.
