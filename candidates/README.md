# Candidates and the bench

Every possession pipeline we built sits behind one interface (`predict`, plus `predict_proba`
for the ensemble) and is scored by one shared bench, `bench.py`. Shipped: `ensemble_ctrl_v5c` and
the folders it needs (`temporal_tcn_v3`, `handover_mstcn_v3`, `ball_radar_v3`,
`learned_dfl_evidence`). Built and dropped: `archive/` (see its README). Analyses:
`stats_report/` (statistical tests), `control_events/` (event-defined control labels).

## Benches

| flag | data | labels | use |
|---|---|---|---|
| (none) | Metrica Sample Games 1 + 2, degraded with `tune_on_sample_data.degrade` x 2 seeds | practice labels from Metrica's event data | secondary check |
| `--holdout` | Metrica Sample Game 3, same degradation | same | run once per final candidate |
| `--dfl` | 7 open DFL / IDSSE Bundesliga games, `tune_on_sample_data.degrade` x 1 seed | DFL tracking flags (BallStatus / BallPossession), same meaning as the competition's | selection until 2026-11-07 |
| `--dfl2` | the same 7 games, `degrade_v2` x 3 seeds | same | **primary selection from 2026-11-08** |

On both DFL benches every model a candidate loads is swapped for its leave-one-game-out fold
model, so a game is only ever scored by models that never saw it (LightGBM boosters and
`torch.load` are swapped automatically, other files go through `dfl_folds.resolve`). Large work
files (DFL data, fold models, caches) live in `$MTAH_WORK`, default `<repo>/external/work`.

## Why bench v2: the simulator did not look like the real tracking

`match_stats.py` computes label-free statistics of a tracking feed. On the competition match
(BAR-ATH, positions only) vs the v1-degraded DFL games:

| | BAR-ATH | DFL, v1 degradation | DFL, `degrade_v2` |
|---|---|---|---|
| ball missing (graded frames) | 0.407 | 0.410 | 0.389 |
| ball gaps per 90 min | 127 | 1779 | 119 |
| gap length p10 / p50 / p90 (s) | 0.5 / 10.7 / 33.8 | 0.2 / 0.9 / 2.8 | 0.5 / 11.2 / 40.8 |
| share of missing frames in gaps > 5 s | 0.997 | 0.090 | 0.972 |
| ball jitter (m) / median acceleration (m/s²) | 0.009 / 5.7 | 0.214 / 363 | 0.007 / 5.6 |
| ball speed p50 / p90 (m/s), not fitted | 5.1 / 14.3 | 9.7 / 18.9 | 4.8 / 14.9 |
| visible players per team | 10.2 / 10.3 | 10.9 / 10.7 | 10.2 / 10.0 |
| tracklets per 90 min, median length (s) | 1036, 35.5 | 7178, 11.4 | 1237, 61.8 |
| player median acceleration (m/s²) | 0.85 | 6.25 | 0.92 |

The v1 simulator hid the ball in many short random bursts during play and added 0.2 m of
noise. The real ball is smooth and missing almost only in long gaps. During those gaps BAR-ATH's
players move like DFL players at dead ball: median team speed 1.0 m/s, against 1.02 dead and
2.27 alive on DFL. A mixture fit of the speed histograms puts about 95% of long-gap time at dead
ball. `degrade_v2.py` therefore hides the DFL ball around the DFL dead-ball runs (plus a few
short open-play gaps), smooths the ball and players, and fragments player tracks into longer
tracklets. Its knobs were fitted to the BAR-ATH statistics with `fit_degrade_v2.py`. Only
positions of the match are used, never a label (RULES.md allows inspecting the data). On the
v2-degraded DFL games the ball is hidden in about 80% of dead-ball frames and about 6% of alive
ones.

## Ranking on bench v2 (2026-11-08)

DFL leave-one-game-out, 7 games x 3 degradation seeds. SE is the standard error over games.
Possessions are per seed, summed over the 7 games (truth: 3061).

| candidate | spellF1 | SE | macroF1 | possessions |
|---|---|---|---|---|
| ensemble_vote | **0.6564** | 0.017 | 0.920 | 2245 |
| handover_mstcn | 0.6515 | 0.016 | 0.917 | 2343 |
| temporal_tcn | 0.6451 | 0.017 | 0.915 | 2385 |
| ball_radar | 0.6311 | 0.017 | 0.915 | 2244 |
| archive/semimarkov_lattice | 0.6257 | 0.015 | 0.910 | 2355 |
| archive/pitch_control | 0.6236 | 0.017 | 0.910 | 2181 |
| archive/ballfree_dead | 0.6167 | 0.017 | 0.911 | 2190 |
| archive/v2_stacked_fixes | 0.6161 | 0.016 | 0.908 | 2190 |
| learned_dfl_evidence | 0.6156 | 0.016 | 0.909 | 2158 |
| archive/skillcorner_domain | 0.6118 | 0.017 | 0.907 | 2137 |
| archive/pathcrf_dynamic | 0.6113 | 0.017 | 0.908 | 2159 |
| archive/hier_touch_decoder | 0.5949 | 0.017 | 0.899 | 2173 |
| archive/v1_original | 0.3697 | 0.011 | 0.786 | 2137 |

The order is close to the v1 bench's. `ball_radar` loses the most (0.647 on v1 to 0.631), as
expected once the ball is missing mainly at dead ball. Paired differences (same game and seed)
are far tighter than the per-candidate SE. The ensemble beats `temporal_tcn` by +0.011 (paired
SE 0.003, 17/21 wins), `ball_radar` by +0.025 (21/21) and `learned_dfl_evidence` by +0.041
(20/21). Against `handover_mstcn` it is +0.005 ± 0.004 (15/21), which is not resolved.

## Error budget and the "controlled possession" finding (2026-11-08)

`error_budget.py` on the ensemble's bench-v2 predictions (21 game-seeds, 9183 true spells):

- **Short spells account for most of the misses.** Recall is 0.03 for true spells under 1 s
  (12.5% of all spells), 0.38 for 1-3 s (30.5%), 0.70 for 3-8 s and 0.91 above 8 s.
- **Why spells are missed:** wrong team 41%, predicted dead 21%, merged into a longer same-team
  spell 22%, boundary off 16%. In 64% of missed spells the ball was visible throughout.
- **Oracle headroom:** true boundaries with our labels +0.072; true labels inside our boundaries
  +0.029; perfect dead ball only +0.014; true sub-1 s spells pasted in +0.085; sub-2 s +0.169.
  Every single model shows the same pattern.

The DFL BallPossession flag flips on every touch. CHALLENGE.md defines a change of possession
differently. It happens only when the opponent gains **controlled** possession: a touch without
control, a deflection that does not intercept, and an aerial duel won without later control do not
count, and the suggested proxy is "a player keeps the ball for a few seconds". `control.fold`
applies that rule. Folding DFL spells shorter than T seconds back into the preceding possession
gives these counts per 90 min:

| T | 0 (raw DFL) | 1 s | 2 s | 3 s |
|---|---|---|---|---|
| possessions per 90 min | 412 | 344 | 272 | 224 |

So part of the apparent under-splitting is an artefact of scoring against every-touch labels.
From 2026-11-08 the bench therefore also reports `ctrl_mean`, the mean spellF1 against controlled
labels with T = 1, 2, 3 s. T is not known, and the definition only bounds it as "a few seconds".
`ctrl_mean` is the selection criterion.

Applying the same rule to a candidate's own output (`ensemble_ctrl`) at CONTROL_S = 2.0 s raises the
ensemble's ctrl_mean from 0.696 to 0.728. 2.0 s had the best mean over T = 1-3 and the best
worst case (a loss of at most 0.048 against the best threshold for any T). It is not statistically
distinguishable from 2.5 s (+0.003 [-0.006, +0.011], p = 0.51), although it does beat 1.5 s
(Holm p = 0.04); see stats_report/. It costs 0.034 against
raw every-touch labels, which the definition rules out.

## Days 4-6: improvements, checks and the freeze (2026-11-08)

All on bench v2, leave-one-game-out. ctrl_mean is the selection metric. "Event labels" are the
DFL possession spells kept only where the new team makes a controlled action (pass, shot, ball
claim, won tackle, carry or a 3 s hold), built from the open DFL event data
(`control_events/`). Paired differences are over 7 games x 3 seeds, SE over games.

| candidate | ctrl_mean | event labels | raw spellF1 | Metrica dev | Metrica holdout |
|---|---|---|---|---|---|
| **ensemble_ctrl_v2 (shipped)** | **0.744** | **0.768** | 0.622 | 0.664 | 0.678 |
| ensemble_ctrl | 0.728 | 0.746 | 0.622 | 0.673 | |
| ensemble_vote | 0.696 | 0.702 | 0.656 | 0.707 | 0.705 |
| handover_mstcn_v2 | 0.727 | | 0.625 | | |
| temporal_tcn_v2 | 0.717 | | 0.643 | | |
| learned_dfl_evidence + fold 2.0 | 0.698 | | | | |
| archive/decision_tree_a (two readable trees) | 0.694 | | 0.582 | 0.645 | |
| archive/v1_original | 0.444 | | 0.370 | 0.475 | 0.506 |

- **Retraining on the realistic degradation** (`make_dataset_v2.py`), with controlled-possession
  targets, is the one Days 4-5 track that clearly helped. `ensemble_ctrl_v2` beats ensemble_ctrl
  by +0.016 (SE 0.003, 21/21) on ctrl_mean and by +0.021 (SE 0.005, 20/21) on event labels, and
  ties on raw labels. Re-run independently: identical codes on all 21 game-seeds.
- **Self-training on the scored game** (archive/selftrain_ctrl) gains +0.004 nested (p = 0.08).
  Not shipped.
- **Decision trees** are interpretable and reach LDE level (0.694), but not the ensemble's.
- **The unknown control threshold matters more than any model choice among the top candidates**
  (stats_report). The event labels (262 possessions per 90 min, between the 2 s and 3 s folds)
  rank candidates like ctrl_mean (Spearman 0.88). Nested choice of the fold prefers
  2.25-2.75 s by only +0.009 ± 0.007, so 2.0 s stays: it has the smallest worst-case loss over all
  label readings.
- **The holdout and dev scores fall** (0.705 to 0.678 on the holdout) because Metrica's practice
  labels count every fine possession, and the control rule merges brief ones on purpose.
- **Match:** 262 possessions, dead share 0.48. The CSV is deterministic (sha256 2de360e9…) and
  regenerated by `make_submission.py`. The one-off public 5-minute correctness check, run after
  the choice was fixed, gave spellF1 0.657 (macroF1 0.880; ensemble_vote 0.619, the original
  pipeline 0.426). It has 17 possessions and was not used for any choice.
- **A label-free warning (stats_report):** a classifier tells the match apart from every DFL game
  (AUC 0.99, mostly player-tracking differences), and the members disagree more on the match (6.2%
  of frames) than on any bench game (3-5%). Expect the real score to be below the bench.

## Round 2 (2026-11-08 to 11-09): the ground truth is finer than we assumed

- **Metrica's own rules** (`control_events/metrica_rules.py`): possession built from Metrica's
  event tagging of its open sample games gives 305–331 possessions per 90, with 17–22% of spells
  under 2 s. That implies T of about 1.2–1.5 s, not 2.
- **The organisers' published shift scores pin it down** (EVALUATION.md: a perfect submission 3 / 8 /
  25 frames late scores 0.982 / 0.893 / 0.654 on the full match). That length fingerprint lies
  between the raw DFL flag and a 1 s fold, and matches Metrica's practice labels; our 2 s target
  gives 0.999 / 0.989 / 0.797. The selection metric became **fine4** (`score_sel2.py`).
- **Selection:** nested choice of member training variant x smoothing x fold on fine4 picked the
  raw-flag members + 1.25 s fold: **`ensemble_ctrl_v4`**, +0.048 fine4 on all 21 game-seeds,
  Metrica dev 0.708 (v2 0.664), holdout 0.707 (v2 0.678). See `ensemble_ctrl_v4/README.md`.
- **Tried, no gain:**
  - `archive/ensemble_ctrl_v3b`: a per-spell control classifier. It adds +0.002 nested on sel2,
    no better than duration alone.
  - Vote smoothing changes.
  - A restart-team rule: the vote is already 96.5% right after a stoppage.
- **`degrade_v3` / `bench --dfl3`:** player tracks break in crowds, as on the match (62% of
  tracklet ends in a crowd vs 16% under v2). The shipped models do not lose on it, so they were
  not retrained.
- **Match disagreement:** the members' excess disagreement on the match (6.3% vs 3–4%) is almost
  all dead vs live while the tracked ball is outside the pitch lines, in brief flickers.
- **Kick-off:** the second half's pre-kick-off frames are now D (FAQ.md), via the "Kick Off
  (Start)" tag.

## Round 3 (2026-11-09): a training-data fix, and a 64-game World Cup bench

- **Bug:** `learned_dfl_evidence/make_dataset.attack_sign` took each game's attack direction from
  the home team's mean x over half 1, which is wrong for DFL J03WR9. The home team kicks off from
  the left at x 44.2 m but averages 54.9 m. The bench (kick-off positions) and the match
  (metadata) were never affected.
- **Fix:** `temporal_tcn_v3`, `handover_mstcn_v3` and `ball_radar_v3` are the same recipes
  retrained on corrected data. In the vote with v4's rule (**`ensemble_ctrl_v5c`**) they give
  fine4 0.717 (+0.008 ± 0.004, 17/21). The bit-identical J03WR9 fold models change by exactly 0,
  so the gain is the fix, not retraining noise.
- **PFF World Cup 2022:** 64 games, labels from PFF's own events (`pff/`). It is held out from
  every choice:
  - v4 beats v2 on every reading except the 2 s fold;
  - v5c beats v4 by +0.006 ± 0.002 (45/64).
- **Metrica:** the holdout (Game 3, once) gives v5c 0.719 vs v4 0.707. Metrica dev prefers v4
  (−0.011, 2 games).
- **Tried, no gain:**
  - `ensemble_ctrl_v5a`: boundary snapping; short-spell insertion from probabilities, MS-TCN
    boundary peaks or touches.
  - `ensemble_ctrl_v5b` + `degrade_v4`: a calibrated dead-ball simulator; dead-ball clean-up
    rules. On the harder bench v4's choices hold, including fold 1.25 s in 7/7 folds.
  - Soft votes, extra members, another decoder.
