# ensemble_ctrl_v5b: the dead-ball call, and a bench calibrated to the match (not shipped over v4)

Two things live here:

1. **A match-calibrated bench, `bench_v4.py` (`--dfl4`)**: the 7 DFL games degraded with
   `candidates/degrade_v4.py`, a dead-ball model refitted to the competition match's label-free
   ball statistics and checked against the organisers' published full-match baseline scores.
2. **`pipeline.py`**: `ensemble_ctrl_v4` unchanged plus one dead-ball clean-up (`JOIN_D_S` =
   2 s: a team spell shorter than 2 s between two D runs becomes D). It is the only dead-ball
   rule that survived nested selection on the calibrated bench, and its gain is at noise level.

**Recommendation: keep `ensemble_ctrl_v4`.** v5b is validated (live == offline, deterministic,
valid CSV) and may be preferred by a team that adopts the calibrated bench as primary, but the
paired evidence is +0.0015 fine4 on the calibrated bench and −0.001 on bench v2, and it changes 7
of 456 spells on the match. Everything else in this track, including the headline question of
whether the bench is too easy, is a finding rather than a gain.

## 1. Is the bench easier than the match? Yes, and most of it cannot be explained label-free

The four `starter/baseline.py` variants (nearest tracked player to the ball; D where the ball is
missing; majority vote over ±0 / 25 / 50 / 125 frames) are published with full-match scores
(EVALUATION.md). The same code on our benches (`_scratch/v5b_calib1.py`, 7 games × 3 seeds; spellF1
against the raw DFL flag, fine4 in brackets):

| baseline | published macroF1 / spellF1 | bench v2 (degrade_v2) | bench v4 (degrade_v4) |
|---|---|---|---|
| raw | 0.6532 / 0.0486 | 0.728 / 0.103 | 0.733 / 0.107 (0.098) |
| smooth25 | 0.6752 / 0.1996 | 0.740 / 0.289 | 0.724 / 0.268 (0.272) |
| smooth50 | 0.6724 / 0.2433 | 0.729 / 0.301 | 0.698 / 0.273 (0.284) |
| smooth125 | 0.6356 / 0.2242 | 0.666 / 0.212 | 0.600 / 0.196 (0.208) |

The gap shrinks with the window (raw: 2× on spellF1, +0.075 on macroF1; smooth125: the match is
actually a little easier), so it is a short-time-scale effect. The label-free statistics of the
match tracking (`_scratch/v5b_calib3/4/6/7.py`) say what that effect is:

- **The ball is visible in short islands during stoppages.** 49 visible runs of 1–10 s per 90
  (v2 bench: 23), 52% of them with the players at dead-ball speed (<1.5 m/s mean; v2: 22%). In
  those islands the ball is inside the pitch (93%), 1.7 m from the nearest player, moving at
  ~2.8 m/s and almost never static (6%). It is a ball carried, bounced or rolled by players during
  the stoppage. The clean DFL feed instead shows a *frozen* ball at the out-of-bounds point (99%
  static) and `degrade_v2` revealed exactly that frozen ball in 22% of dead runs: a dead cue the
  match never shows.
- **A lost ball stays lost ≥ 5 s.** Gaps of 0.5–5 s: 1.7 per 90 on the match vs 22 on bench v2;
  gaps of 5–10 s: 45 vs 18 (stoppages cut into pieces by the islands). Gaps 10–30 s are at least
  as dead-like by player speed as the v2 bench's (P(speed < 1.5) 0.80 vs 0.67), so the gaps are
  not open play: the FAQ's "ball missing in duels and crowded boxes" is a small part of the 41%.
- **A ball going out is tracked ~0.8 s past the line while moving** (69 out-runs per 90, median
  0.84 s, 10% static); v2 hid it 2.2 s *before* the dead flag. Consequence (coordinator's point
  3): bench v2 hid the ball within ±0.5 s of 12.9% of true possession changes (10.4% exactly at
  the change); the clean feed never does, and the match tracker has the ball at 97% of v4's
  predicted changes. degrade_v4 hides it at 3.2%.
- **Checked and ruled out as the cause of the macroF1 gap** (all label-free): nearest-player
  contestedness (both teams within 1 m of each other at the ball: 0.172 on the match, 0.137–0.194
  on DFL games), team-count imbalance (P(|home−away visible| ≥ 3) 9% vs 6%), a ball-vs-player
  time lag (the shift test peaks at +3…+6 frames on both the match and the bench), kick-off
  onsets (the ATD ball starts moving at frames 11188 and ~113735 against the published anchors
  11185 / 113742), ball static share (5% vs 2.4% in clean alive play: almost no parked ball), gap
  geometry (gaps whose vanish and reappear points are both inside the pitch, 20 m apart and with
  fast players — open-play losses — are 4.5% of the match's 112 gaps ≥ 5 s, about 5 per 90).

**What degrade_v4 does** (per dead run): the ball rolls on (extrapolated, 0.6 s mean, 40% of
runs), is hidden, reappears in Poisson islands (2 per minute of hidden dead time, lognormal
3.5 s) carried by a player chosen near the restart spot (offset 1.5 m, random walk), is carried
to the spot for 2.5 s and parked (clipped to the line) 0.6 s before the restart; open-play gaps are
≥ 4.5 s (15 per 90); gaps under 5 s are bridged by interpolation as the match tracker does.
Players are degraded as in v2. The knobs were set from the run statistics above
(`fit_degrade_v4.py --check` prints the comparison; a random search, `--evals`, did not find this
basin and `--structure-only --free-all` converged on degenerate corners, see
`$MTAH_WORK/arch/ensemble_ctrl_v5b/fit_v4_run*.log`). Result: gap histogram 0 / 42 / 62 / 14 per
90 (match 1.7 / 45 / 51 / 16 for 0.5–5 / 5–10 / 10–30 / 30+ s), visible-run histogram 7.6 / 16.8 /
20.3 / 39.4 (match 9.3 / 17 / 22.9 / 42.5), island ball speed 2.9 m/s (2.84), ball missing 0.37
(0.41), static 4% (5%).

**What it does not do:** reproduce the published macroF1 level. With realistic dead-ball
structure the raw baseline stays at macroF1 ≈ 0.73 on any DFL degradation; the published 0.653
would need the nearest-player rule to be right on only ~57% of alive visible frames (it is
0.735 on bench v2, 0.765 on clean DFL) or a dead share near 0.55 with a third of it visible. The
uncapped random fit reached 0.689 only by hiding the ball in 40 open-play gaps of 6–12 s per 90,
which the gap geometry and player speeds exclude (`fit_v4_run1.log`). So: part of the
"bench easier than match" gap is the dead-ball structure above, part is the truth's finer
granularity (fine4 already averages over it), and a residual of ~0.03–0.07 macroF1 on the
unsmoothed baselines has no label-free explanation we could find. It could be alignment or
label semantics of the private truth; it cannot be validated off-match.

**Public window, read from DATA.md only (not used for any choice):** the first 5 minutes have the
ball missing on 11.8% of frames while DATA.md lists D on 19.6% of them, so at least 40% of that
window's dead time has a visible ball; whole-match missing is 40.7% and varies from 12% to 80% by
5-minute block, tracking the players' slow share (dead time is bursty: long stoppages late in
each half).

## 2. Does the calibrated bench change v4's choices? No

v4 re-run live on bench v4 (`bench_v4.py candidates/ensemble_ctrl_v4/pipeline.py`, 21 game-seeds,
`$MTAH_WORK/probs/ensemble_ctrl_v5b_v4_dfl4`): fine4 0.6956 (v2 bench 0.7095), raw 0.660,
sel2 0.720, macroF1 0.919. Harder, as intended. The members (`pipeline_raw.py` variants and
ball_radar) were re-run on it too (`$MTAH_WORK/probs/ensemble_ctrl_v5b_members/`) and the
nested fold × joinD choice (`_scratch/v5b_nested.py dfl4`) picks fold 1.25 s in 7/7 folds, as on
bench v2; fold 2 s costs −0.039 fine4 on both benches.

v4's dead calls on bench v4 vs truth: 94.6 vs 95.8 D runs per 90, dead share 0.434 vs 0.449, 2–5 s
D runs 4.8 vs 6.9 per 90. On the match v4 writes 113.9 D runs per 90 with 17.8 of them 2–5 s
long (52% with the ball visible). degrade_v4 does **not** reproduce that excess: the members
handle carried-ball islands correctly, so the match's short D runs come from something else
(the members' match disagreement sits where the ball is tracked outside the lines while moving;
round2_notes) or are real quick restarts. Whether they are errors cannot be validated off-match.

## 3. Dead-ball rules, nested on fine4 (`_scratch/v5b_rules.py`)

Grid: minimum D length (0–4 s, short D handed to the surrounding / following team), join D runs
separated by a team spell < T (0–4 s), the same restricted to D runs with the ball visible inside
the pitch, or with fast players. Picked on 6 games, scored on the 7th, 3 seeds, vs v4's codes:

| rule | bench v2 nested | bench v4 nested | in-sample best |
|---|---|---|---|
| any minD, visD, fastD | picked 0 (no-op) in 7/7 | picked 0 in 7/7 | all ≤ 0 |
| joinD 2 s | −0.0011 ± 0.0011 (0/7 games) | **+0.0015 ± 0.0004** (6/7 games, 13/21 game-seeds) | +0.0015 |
| joinD 3 / 4 s | −0.005 / −0.009 | −0.000 / −0.003 | |

Why the sign flips: joinD 2 s removes 1.1 spells per 90 on bench v2, 96% of them truly alive
(quick restarts), and 1.6 per 90 on bench v4, 83% truly dead (carried-ball islands). On the match
it removes 7 spells of 456 (5.9 per 90; ball visible in 76% of their frames, median 1.7 s). The
match has the islands, so the bench-v4 reading is the better model of it, but the stakes are
< 0.01 spellF1 either way. The coordinator's other rules (minimum D 2 s, D under 3 s inside one
possession, goal = long gap ending at the centre spot) are covered by minD / the fold / the
kick-off tag and were not picked.

## 4. Results (live runs)

| | fine4 | sel2 | raw | macroF1 | poss (21 game-seeds) |
|---|---|---|---|---|---|
| ensemble_ctrl_v4, bench v2 | 0.7095 | 0.7409 | 0.6737 | 0.9272 | 6691 |
| **ensemble_ctrl_v5b, bench v2** | 0.7089 | 0.7403 | 0.6731 | 0.9270 | 6666 |
| ensemble_ctrl_v4, bench v4 | 0.6956 | 0.7204 | 0.6602 | 0.9189 | 7192 |
| **ensemble_ctrl_v5b, bench v4** | 0.6971 | 0.7218 | 0.6616 | 0.9192 | 7157 |

- **Paired vs v4** (`score_sel2.py --ref`, SE over 7 game means): bench v2 fine4 −0.0006 ± 0.0004
  (5/21 game-seeds), sel2 −0.0007 ± 0.0003; bench v4 fine4 **+0.0015 ± 0.0004** (13/21), sel2
  +0.0014 ± 0.0003. The nested selection above gives the same numbers because joinD 2 s was
  picked in every fold on bench v4 and never on bench v2.
- **Live == offline:** the live codes equal joinD 2 s applied to v4's saved codes on 21/21
  game-seeds of bench v2 and 21/21 of bench v4 (`np.array_equal`).
- **Metrica dev** (Sample Games 1 + 2, practice labels): spellF1 0.7094 (v4 0.7084), macroF1
  0.825. Holdout (Game 3) not run: the candidate is not proposed for shipping.
- **Match:** 315 possessions (v4 322), mean 10.5 s, dead share 0.478 (v4 0.476), no X inside the
  halves, `validate_submission.py` passes; two `--match` runs are `cmp`-identical
  (sha256 e8c21831…; v4 2ea055d4…), 294 frames differ from v4. 7 spells of v4's are merged into stoppages; nothing else changes.

Reproduce:

```
OMP_NUM_THREADS=2 python candidates/bench.py candidates/ensemble_ctrl_v5b/pipeline.py --dfl2 \
    --folds $MTAH_WORK/arch/ensemble_ctrl_v2/folds.json --save-probs $MTAH_WORK/probs/ensemble_ctrl_v5b
python candidates/score_sel2.py $MTAH_WORK/probs/ensemble_ctrl_v5b --ref $MTAH_WORK/probs/ensemble_ctrl_v4
OMP_NUM_THREADS=2 python candidates/ensemble_ctrl_v5b/bench_v4.py candidates/ensemble_ctrl_v5b/pipeline.py \
    --folds $MTAH_WORK/arch/ensemble_ctrl_v2/folds.json --save-probs $MTAH_WORK/probs/ensemble_ctrl_v5b_dfl4
python candidates/score_sel2.py $MTAH_WORK/probs/ensemble_ctrl_v5b_dfl4 --split dfl4 --ref $MTAH_WORK/probs/ensemble_ctrl_v5b_v4_dfl4
python candidates/fit_degrade_v4.py --check          # degrade_v4 vs the match's statistics and the published baselines
python candidates/bench.py candidates/ensemble_ctrl_v5b/pipeline.py --match
```

Files: `pipeline.py` (v4 + `dead_cleanup`), `members.py` / `ensemble_core.py` (copies of v4's),
`bench_v4.py` (the dfl4 bench; cache prefix `dfl4_`), `candidates/degrade_v4.py`,
`candidates/fit_degrade_v4.py`, analyses in `candidates/_scratch/v5b_*.py`.
