# decision_tree_b

This folder does two things:

1. **An audit of `candidates/decision_tree`.** It checks how that tree was evaluated and whether
   its interpretability claims hold. The audit is read-only on the teammate's files.
2. **A rule model small enough for one slide.** A dead-ball tree with 8 leaves and a team
   scorecard with 8 point values, run through the same decoder and `control.fold(2.0)`.

All scores are DFL bench v2 (`bench.py --dfl2`): 7 games x 3 degradation seeds, leave one game
out. Every game is scored by fold models that never saw it. SE is over the 7 games. "Paired" means
the same game and seed, averaged over seeds, with the SE taken over games.

## Headline

| | decision_tree (teammate) | decision_tree_b (rules) | paired difference |
|---|---|---|---|
| ctrl_mean, with the 2 s fold | 0.6661 | **0.6541** | -0.012 ± 0.005 (rules win 2 of 7 games) |
| spellF1 against raw DFL labels | 0.5595 | 0.5437 | -0.016 ± 0.006 (1 of 7) |
| macroF1 | 0.8984 | 0.8895 | -0.009 ± 0.001 (0 of 7) |
| ctrl_mean without the fold (`pipeline_raw.py`) | 0.6108 | 0.6404 | **+0.030 ± 0.009** (6 of 7) |
| as the third voter with temporal_tcn + handover_mstcn, then fold 2 s | 0.7275 | 0.7267 | -0.001 ± 0.002 (3 of 7) |
| Metrica dev spellF1 | 0.6358 | 0.6278 | - |
| match (`--match`) | 342 possessions, 45.7% dead | 288 possessions, 47.1% dead | - |
| model size | 1885 nodes, 141 features | 8 leaves + 8 point values, 18 inputs | - |

**The slide-sized model keeps 98% of the tree's ctrl_mean** (0.654 against 0.666). That is 88%
of the gap between a depth-3 tree (0.566) and the depth-12 tree. Without the control fold it is
clearly better than the tree (+0.030), because a smooth scorecard flickers less than
piecewise-constant leaves. The fold then removes most of the tree's flicker.

## Audit of decision_tree, most important first

Script: `audit.py`. Output: `$MTAH_WORK/arch/decision_tree_b/audit/{audit.json,audit.log}`.

### 1. "Three features carry ~80% of importance" is a training-set gini artefact, and it ranks the wrong features

- **Gini attribution is unstable.** It was measured on the full tree's training data. On the 7
  fold trees the gini importance of `ball_disp_b` is 0.49-0.51 in 6 folds and **0.02 in fold 2**.
  There the root split is on `ball_line` instead, with gini 0.49. Both features are NaN exactly
  when the ball is untracked, and the NaN-left split is what does the work. So the 0.51 belongs to
  "is the ball tracked?", not to "how far did the ball move?".
- **Permutation importance on held-out games** (fold tree gi on game gi, graded frames, 3 seeds).
  Held-out frame accuracy is 0.913.

  | feature | gini (mean of folds) | permutation drop in accuracy (SE) | permuted, missingness kept |
  |---|---|---|---|
  | ball_disp_b | 0.437 | 0.179 (0.011) | **0.053** |
  | lastctrl | 0.227 | 0.058 (0.012) | 0.057 |
  | any_since_ctrl | 0.067 | 0.035 (0.007) | 0.034 |
  | mean_speed_m0.5 | 0.006 | 0.017 (0.012) | 0.017 |
  | mean_speed_s-0.8 | 0.009 | 0.016 (0.005) | 0.016 |
  | ball_xdir_pmf | 0.013 | 0.016 (0.001) | 0.014 |
  | ball_line | 0.078 | 0.015 (0.004) | 0.014 |

  The top three take 59% of the positive permutation importance, not 80%. About 70% of
  `ball_disp_b`'s importance disappears when its NaN pattern is kept and only the values are
  shuffled. What the tree learned is mainly "an untracked ball is dead".
- **On the selection metric the ranking changes.** This test permutes one input in the tree and
  runs the full decoder plus the 2 s fold (base ctrl_mean 0.6661):

  | input | ctrl_mean drop (SE, games that got worse) |
  |---|---|
  | ball_xdir_pmf (ball moving towards a team's goal over 2 s) | **0.037** (0.007, 7 of 7) |
  | ball_disp_b | 0.028 (0.005, 7 of 7) |
  | lastctrl | 0.013 (0.006, 6 of 7) |
  | ball_line | 0.010 (0.006, 7 of 7) |
  | any_since_ctrl | 0.008 (0.004, 4 of 7) |
  | ball_speed, mean_speed_*, ball_out_m2.0 | ≤ 0.002, not significant |

  The feature that matters most for the score has 1% of the gini. Gini is dominated by the many
  easy dead-ball frames. The score depends on possession boundaries, and the decoder's own
  long-gap term already covers most of the missing-ball rule.
- **Caveat on the rule itself.** On the bench, "ball missing means dead" is partly built into
  the simulator. `degrade_v2` hides the ball around DFL dead-ball runs on purpose (about 80% of
  dead frames are hidden, about 6% of live ones). That design rests on the BAR-ATH mixture fit
  (about 95% of long-gap time is dead), not on labels. The rule is only as true on the match as
  that fit is.

**Should say:** "On held-out games, 'is the ball tracked' (`ball_disp_b`/`ball_line` NaN) and
the last controlling team carry most of the per-frame accuracy. For the selection metric, the
ball's direction of travel (`ball_xdir_pmf`) matters most, then the ball-missing split, then
`lastctrl`. Gini shares are not stable across folds."

### 2. "As an ensemble member it adds nothing" is not supported

Setup: frame majority vote of temporal_tcn + handover_mstcn + X, ties go to temporal_tcn, then
fold 2 s. All runs use saved bench-v2 codes.

| third voter | ctrl_mean | vs tree |
|---|---|---|
| decision_tree (raw codes) | 0.7275 | - |
| learned_dfl_evidence | 0.7252 | tree +0.0024 ± 0.0029 (4 of 7 games), **not resolved** |
| ball_radar: saved `ensemble_ctrl` codes (its vote also merges sub-1 s spells), then fold | 0.7283 | tree -0.0007 ± 0.0022 (4 of 7) |
| none (= temporal_tcn + fold) | 0.7203 | tree **+0.0073 ± 0.0029 (6 of 7)** |
| decision_tree_b (raw) | 0.7267 | -0.0008 ± 0.0018 (3 of 7) |

The README's numbers reproduce exactly. The tree adds as much as any third voter, and it cannot
be told apart from LDE, ball_radar or the rule model. **Should say:** "interchangeable with LDE
and ball_radar as the third voter (+0.007 over the two-model vote). It is no better, and it
disagrees with LDE on only 5.1% of graded frames."

### 3. Leakage from selection: small, and in the conservative direction

- **Nested depth and leaf selection.** For each game, the setting best on the other 6 is scored
  on the 7th, over all 11 settings. L100_d12 wins on every 6-game subset, so the nested estimate
  is **0.6681**, which is in-sample best and 0 optimism. The reported setting, L200_d12 at 0.6661,
  is the runner-up (L100_d12 - L200_d12 = +0.0020 ± 0.0024). The reported 0.666 is therefore
  not inflated by the depth choice.
- **Control threshold T, nested** (the tree's raw codes, re-folded). Each fold picks 2.5 s,
  giving 0.6699. 2.5 - 2.0 = +0.004 ± 0.006, not significant. The fixed 2.0 s was chosen for
  the ensemble on the same 7 games, so it is shared selection on the test games, but for this
  tree it is slightly pessimistic.
- **Choices made on all 7 games and never nested.** The decoder constants (LDE round 3:
  mostly Metrica, a few checked on DFL LOGO), `CONTROL_S`, and the feature set. These are shared
  by every candidate, so they do not bias comparisons between candidates. They do bias every
  absolute number by an unknown, probably small, amount.
- **The depth sweep in the summary mixes leaf sizes.** Depths 3-12 use leaf 200, while 16 and
  20 use leaf 100. Like for like: L100 d12 / d16 / d20 = 0.668 / 0.665 / 0.663.

### 4. Significance of the depth differences (paired over games, ctrl_mean)

| comparison | difference | SE | games won |
|---|---|---|---|
| d5 - d3 (L200) | +0.031 | 0.005 | 7 of 7 |
| d8 - d5 (L200) | +0.045 | 0.006 | 7 of 7 |
| d12 - d8 (L200) | +0.024 | 0.008 | 6 of 7 |
| d12 L200 - d12 L2000 | +0.020 | 0.006 | 7 of 7 |
| d12 - d16 (L100) | +0.003 | 0.002 | 5 of 7 |
| d16 - d20 (L100) | +0.002 | 0.002 | 4 of 7 |

The gains up to depth 12 are real. The plateau from 12 to 20 is within noise. The claim
"deeper fragments the output" holds for the raw decode. Against raw labels with the fold,
depth 20 is actually best (0.5666 against 0.5614 for L100 d12).

### 5. Fold files are correct

- Each fold tree's root holds 2 x (all rows - rows of game gi), exactly. For gi = 0..6:
  1028010, 1033298, 1038512, 1032128, 1027718, 1032356, 1026998. The full tree holds
  1203170 = 2 x 601585. Game sizes differ, so each count identifies the excluded game.
- `folds/fold_tree_<gi>.npz` is byte-identical to `sweep/L200_d12/fold_tree_<gi>.npz`.
  `folds.json` maps sha256 prefix `642af161de9b`, which is `tree.npz`. The bench log shows
  `tree.npz -> fold_tree_0..6`.
- **Training data and bench are disjoint by seed.** `ds_v2` uses seeds 41-43 with
  `default_rng([seed, game_key])` and randomised knobs. The bench uses
  `default_rng(2027..2029)` with default knobs. They share the 7 games, which is standard
  leave-one-game-out, and the held-out game is never in a fold's training set.

### 6. A stale-feature bug, latent in decision_tree/pipeline.py

`features()` caches on `(id(players), id(ball), len, sign, blocks)`. Once an array is freed, the
next game's array can reuse its id. Two seeds of the same game have the same length, sign and
blocks, so the cache can return the previous seed's features. This bit my first sweep, which
called the same helper from long-lived pool workers. 5 of 21 game-seeds were scored on another
seed's features (up to ±0.02 per row). A fresh recomputation matched the bench. The teammate's
bench run and feature cache are not affected: all 21 rows of the rescored L200_d12 match its
bench log exactly. Any process that calls `predict` on several games, such as an ensemble
loader, is exposed. **Fix** (applied here): key the cache on a content hash of ball and players.

### 7. Smaller issues

- `sweep.py` overwrites `sweep_results.json` on each run. The leaf 200 and 2000 rows that the
  README quotes had no per-game record left on disk. I re-scored all 11 settings from the fold
  trees (`audit/sweep_all.json`), and every mean in the README reproduces to 4 decimals.
- The printed rules show `lastctrl <= 0.5 -> A` in the full tree, which puts lastctrl = 0 with
  away, but `lastctrl <= -0.5` in fold 0, which puts it with home. The symmetric averaging
  hides this. It is harmless, but the printed rule is one-sided.
- **The quoted rules do hold on held-out games** (README thresholds, raw DFL labels, all graded
  bench frames):

  | rule | claimed coverage / purity | held-out coverage / purity [per-game range] |
  |---|---|---|
  | ball missing or moved ≤ 0.58 m in 0.5 s -> D | 0.505 / 0.86 | 0.526 / 0.851 [0.83-0.87] |
  | ... and no control for > 3 s, players ≤ 2.24 m/s, ball ≤ 7.9 m/s -> D | 0.405 / 0.98 | 0.423 / 0.977 [0.97-0.98] |
  | ... and control within 3 s, but ball out over 2 s -> D | 0.016 / 0.83 | 0.009 / 0.896 |
  | ball moving: the last controlling team has it | 0.495 / 0.82 | 0.474 / 0.857 [0.81-0.88] |
  | ... that team controlled within 3.4 s | 0.416 / 0.88 | 0.420 / 0.893 |
  | ... and its nearest player is > 1.25 m closer (2 s mean) | 0.282 / 0.94 | 0.286 / 0.946 [0.92-0.96] |

  The top-3 levels are also stable across folds. The root threshold is 0.55-0.60 m and the
  control window is 3.0-3.3 s, except fold 2's root on `ball_line`.

## The rule model (slide)

`python candidates/decision_tree_b/explain.py` prints the following. All numbers were learned
on the 7 DFL games for the shipped `rules.npz`, and inside each fold for the bench.

**Dead ball.** A tree of depth 3 on 10 whitelisted inputs, giving P(dead) per leaf:

```
ball untracked or moved <= 0.58 m in 0.5 s
  someone had the ball alone within 3.0 s
    ball outside the pitch (2 s mean)         -> dead 0.83   ( 1.6% of frames)
    otherwise                                 -> dead 0.23   ( 5.8%)
  nobody had it for > 3.0 s
    players' mean speed <= 2.1 m/s            -> dead 0.97   (40.6%)
    players faster                            -> dead 0.55   ( 2.5%)
ball moving (> 0.58 m in 0.5 s)
  ball inside the pitch
    players nearly still (2 s mean <= 0.9 m/s) -> dead 0.76  ( 0.4%)
    otherwise                                 -> dead 0.01   (48.3%)
  ball outside the pitch
    moving <= 1.42 m in 0.5 s                 -> dead 0.97   ( 0.6%)
    faster                                    -> dead 0.40   ( 0.2%)
```

**Which team, in play.** A points scorecard (logistic, no intercept). Every input flips sign when
home and away swap, so the card is exactly symmetric. P(home) = 1 / (1 + e^-points).

| input (missing counts 0) | points for home per unit | clip |
|---|---|---|
| lastctrl: +1 home / -1 away was last alone on the ball (≤ 1 m, opponent > 1.5 m, 3 frames) | -0.41 | ±1 |
| lastctrl_m2.0: the same, averaged over ±2 s | +0.57 | ±1 |
| since_ctrl_gap: log(frames since away's control) - log(frames since home's) | +0.49 | ±5 |
| nearest_gap_m2.0: home's nearest-player distance minus away's, 2 s mean (m) | -0.12 | ±10 |
| ball_xdir_pmf: ball x towards home's attacking goal, last 2 s minus next 2 s (m) | -0.05 | ±20 |
| fwd_pmf: change in home-minus-away forward speed, last 2 s minus next 2 s (m/s) | -0.15 | ±5 |
| width_gap: home width minus away width (m) | +0.05 | ±30 |
| spread_gap_m5.0: home spread minus away spread, 5 s mean (m) | +0.02 | ±10 |

In words:
- "Whoever had the ball alone most recently has it, more so if that is true across ±2 s."
- "The team closer to the ball, the team the ball is travelling towards the far goal of, and
  the wider team get a little extra."

The negative weight on the raw `lastctrl` is a correction for `lastctrl_m2.0`. When the flag has
been steady for ±2 s, the two together give +0.16. Around a change, the centred average leads
the flag, so the card starts moving towards the next team before `lastctrl` flips. The
recency gap is the largest single term.

**Stability of the learned numbers across folds.** The dead root threshold is 0.55-0.61 m. The
scorecard points move by ≤ 0.07, except the fold without game 6, which moves by up to 0.12.

Then the pipeline applies LDE's decoder (`decoder.py`, a verbatim copy of
decision_tree/decoder.py), followed by `control.fold(codes, 2.0)`. It is the same as
decision_tree, so the two candidates differ only in the evidence model.

## How this model was chosen (and what that costs)

Every setting was scored by leave-one-game-out with fold-trained models (`sweep.py`). The order
of the experiments was:

| family | best setting | ctrl_mean |
|---|---|---|
| two shallow trees, whitelist inputs (D = dead depth, T = team depth) | D4_T4 (D3_T4 0.590, D2_T2 0.566) | 0.591 |
| shallow dead tree + team tree on all ~190 inputs, depth ≤ 6 | D4_T6 (D3_T6 0.619, D3_T4 0.602) | 0.620 |
| **depth-3 dead tree + team scorecard** | D3_TS | **0.654** |

- **The dead stage was never the problem.** A swap test on seed 2027 with fold models and the
  whitelist D4_T4 stages:

  | P(dead) from | P(home) from | ctrl_mean |
  |---|---|---|
  | tree | tree | 0.659 |
  | tree | shallow team stage | 0.587 |
  | shallow dead tree | tree | 0.652 |

  So the shallow dead tree on 10 inputs costs about 0.006, and everything else is in the team
  call.
- **Team call.** Shallow team trees lose 0.05-0.08, because their piecewise-constant
  probabilities give the decoder nothing to work with near a hand-over. The scorecard's inputs
  came from the features the depth-6 all-input team tree split on most often (ball_xdir_pmf,
  width, lastctrl_m2.0, spread, fwd_pmf), plus the whitelist. That tree was trained on all 7
  games, so the input list carries a mild selection leak. The family itself (tree against
  scorecard) was also picked on these 7 games. The gap to the next family (+0.034 ± 0.006, 7 of 7 games)
  is much larger than any plausible optimism. Within the scorecard nothing was tuned: C = 1 and
  the clips were set once.
- The shallow-tree grids were first scored while the id-cache bug (audit item 6) was present.
  Re-scored without it (D2_T2, D3_T3, D3_T4, D4_T4, tall D3_T4/D3_T6/D4_T6, D3_TS), the means
  are identical except D3_TS, which goes from 0.6551 to 0.6541. 0.6541 matches the bench.

## Determinism

- Inference is plain numpy: two tree walks and one dot product. There is no scikit-learn and
  no randomness.
- Two `--match` runs give byte-identical `submission.csv` (`cmp` passes), sha256
  `ef50b82a5adb6f36a9645c1a76b48c28767a788fba9f5ddcdd9da8ec8be0b4ef`.
- Two `--dfl2` runs (before and after the cache fix) give the same ctrl_mean, 0.6541, and the
  same per-T scores. A fresh single-process recomputation of one game-seed reproduced the
  bench's saved codes exactly.
- `train.py` and `train_scorecard.py` use `random_state=0` and deterministic lbfgs. The numpy
  walker matches sklearn `predict_proba` exactly (max |diff| 0).

## Rebuild

```sh
PY=$MTAH_WORK/venv/bin/python; W=$MTAH_WORK/arch/decision_tree_b   # scikit-learn 1.9.1: training only
OMP_NUM_THREADS=2 $PY candidates/decision_tree_b/train.py --jobs 2                 # whitelist trees, D/T 1-4 (~1 min)
OMP_NUM_THREADS=2 $PY candidates/decision_tree_b/train.py --jobs 2 --grow 4 --team-grow 6 --team-all \
    --out $W/sweep_tall                                                            # (diagnostic, ~3 min)
OMP_NUM_THREADS=2 $PY candidates/decision_tree_b/train_scorecard.py --dead D3      # shipped family (~10 s)
OMP_NUM_THREADS=1 $PY candidates/decision_tree_b/sweep.py --jobs 2 --dir sweep_card   # also --dir sweep / sweep_tall
cp $W/sweep_card/D3_TS/rules_full.npz candidates/decision_tree_b/rules.npz
mkdir -p $W/folds; for g in 0 1 2 3 4 5 6; do cp $W/sweep_card/D3_TS/fold_rules_$g.npz $W/folds/; done
# folds.json: {"<sha256(rules.npz)[:12]>": ["$W/folds", "rules"]}   (currently 04859280cac6)
$PY candidates/bench.py candidates/decision_tree_b/pipeline.py --dfl2 --folds $W/folds.json \
    --save-probs $MTAH_WORK/probs/decision_tree_b
$PY candidates/bench.py candidates/decision_tree_b/pipeline_raw.py --dfl2 --folds $W/folds.json \
    --save-probs $MTAH_WORK/probs/decision_tree_b/raw
$PY candidates/bench.py candidates/decision_tree_b/pipeline.py --match            # dev score + submission.csv
$PY candidates/decision_tree_b/explain.py --folds $W/folds                        # the slide
OMP_NUM_THREADS=1 $PY candidates/decision_tree_b/audit.py --jobs 2                # the audit (~3 min)
```

Logs are in `$W`: `train*.log`, `sweep*.log` (the `*_membug` files are the first runs, before the
cache fix), `bench_dfl2{,_raw}.log`, `bench_match{1,2}.log` and `audit/`.

## Files

- `pipeline.py`: the candidate (`predict`, `predict_proba`). `pipeline_raw.py`: the same
  without the fold.
- `train.py`: shallow stage trees. `train_scorecard.py`: the team scorecard. `sweep.py`: LOGO
  sweep with nested estimate.
- `audit.py`: the audit above. `explain.py`: the slide.
- `decoder.py`, `tree_walk.py`: verbatim copies from `decision_tree/`.
- Features come from `learned_dfl_evidence/features.py`, the same code that built `ds_v2`.
