# decision_tree_a

A review and rework of `decision_tree` (the teammate's single 3-class tree). It is still made of
plain, readable decision trees with no forests or boosting: one tree for dead ball and one for
the team call. The team tree gets a few window features it can actually use, and its team
log-odds are smoothed over ±1 s before LDE's decoder.

**DFL bench v2 ctrl_mean 0.6945**, against 0.6661 for `decision_tree`. The paired difference is
+0.028 (SE 0.004), and the new model wins 20 of 21 game-seeds and all 7 games.

## Critique of decision_tree, ranked by measured impact

All numbers are DFL leave-one-game-out (7 games x 3 degrade_v2 seeds, fold trees), ctrl_mean
unless stated. Paired SEs are over the 21 game-seeds.

1. **The evidence flickers, and nothing in the model or decoder integrates it.** Leaf
   probabilities are piecewise constant, so a single threshold crossing flips the team. On alive
   frames the tree's argmax team flips **13.2 times per alive minute; the truth flips 6.4
   times**. LDE's decoder was tuned for smooth boosters. Its local-contrast term (`contrast_beta`
   1.25) even amplifies short dips, and turning it off alone gives +0.007 on the old tree. A
   centred ±1 s moving mean of the team log-odds before the Viterbi gives **+0.0147 (SE 0.0026,
   18/21)** on the teammate's own tree, without retraining. Smoothing by ±0.5 s gives +0.007,
   ±1.5 s +0.013, ±2 s +0.004, and ±3 s -0.030.
2. **The team call has no usable memory feature.** A tree cannot integrate over time. The only
   "who had the ball" memory in LDE's features is the strict confident-control event (player
   within 1 m, opponent beyond 1.5 m, 3 frames), which is rare under degraded tracking. The
   centred moving means (`diff_d1_m*`) mix in-flight frames. `extra.py` adds a looser "near"
   event (ball seen, nearest player within 2 m and at least 0.5 m closer than the other team's)
   and its backward and forward integrals:
   - `lastnear` / `nextnear` and the time since / until each team's near event;
   - `nearbal_b/f{1,3,8}s`: the mean of near_H - near_A over the previous or next window;
   - `margin_b/f{1,3,8}s`: the mean closest-player margin over the frames where someone is
     within 3 m of the ball.

   Same split setup, same decoder: **+0.0199 (SE 0.0043, 20/21)**. The retrained team tree takes
   `nearbal_b3` as its root, with 71% of its gini decrease.
3. **One 3-class tree spends depth on both tasks.** Splitting it into a dead tree and a team tree
   did *not* help by itself. A team tree on the old features scores -0.0063 (SE 0.0029) against
   the joint tree, both smoothed. Splitting is still worth it for readability. A depth-8 dead
   tree (225 nodes, 0.6937) matches the joint tree's P(D) (0.6932), and dead-ball F1 is 0.967 for
   both. Each tree can now be read alone.
4. **The decoder knobs are mostly not knobs.** `tau` is a no-op, because switch and dead costs
   are scaled with tau, so tau 0.10 / 0.15 / 0.25 give the same 0.6661. Once the evidence is
   smoothed, switch_cost 0.4 / 0.75 / 1.5, dead_cost 3 / 5 / 8, contrast_beta 0 / 1.25 / 2.5,
   P(D) smoothing and control_s 2.5 all fall within ±0.003. They were left at LDE's values.
5. **Calibration is not the problem.** Out of fold, p_home on alive frames has ECE 0.011 (log
   loss 0.282), and p_dead on graded frames has ECE 0.005. There is mild overconfidence at the
   extremes: mean p 0.026 in the 0-0.1 bin against a frequency of 0.035. A monotone recalibration
   would barely move the Viterbi, so it was not pursued. The issue is temporal, not marginal.
6. **Latent bug in the feature memo.** `decision_tree/pipeline.py` keys its feature cache on
   `id(players)`, `id(ball)`, the length, sign and blocks. Seeds of one game have the same length,
   sign and blocks, so once the old arrays are freed, a new array can get the same id and receive
   stale features. In my first sweep, which called the pipeline from a fresh function per
   game-seed, this cost 0.003-0.005. bench.py is unaffected today only because it keeps the old
   arrays alive while it loads the next ones. decision_tree_a keys the cache on a sha1 of the
   array contents.
7. **Smaller points, checked but not changed:**
   - **H/A symmetry is correct.** The swap augmentation and the averaging over the swapped
     features use `features.swap_index`. The `kinds` were checked: ball_xdir, diff_* and lastctrl
     are anti; box_* and the per-team blocks are swapped.
   - **Class balance is fine.** 45% of training rows are D, and the swap makes H/A exactly
     balanced.
   - **Leaves are smaller than they look.** Each training frame appears as up to 3 near-duplicate
     degradations (seeds 41-43), and doubled again by the swap. So min_samples_leaf 200 is
     roughly 30-70 distinct moments, and rows 0.2 s apart are strongly correlated. I did not
     retrain at other strides or seed mixes to test this: a fit takes about 50 s and the compute
     budget went to points 1-3. The fold scores show no overfit at depth 10-14 (plateau below).
   - **No hand-over model.** H<->A boundaries still use the first-hit snap. This was not tested.

## Variants (DFL --dfl2, LOGO fold trees, ctrl_mean; possessions per seed, truth 3061)

`old` is the teammate's tree (`arch/decision_tree/sweep/L200_d12`). Unless stated, the decoder is
LDE's, followed by `control.fold(2.0 s)`. `hw` is the half width, in frames, of the team-logit
moving mean.

| dead evidence | team evidence | decoder | ctrl_mean | raw spellF1 | poss |
|---|---|---|---|---|---|
| old (joint) | old (joint) | as shipped | 0.6661 | 0.5595 | 1876 |
| old | old | contrast_beta 0 | 0.6734 | 0.5590 | 1806 |
| old | old | hw 6 / 12 / 25 / 37 / 50 / 75 | 0.6693 / 0.6735 / **0.6808** / 0.6795 / 0.6700 / 0.6364 | 0.5683 (hw 25) | 1813 |
| old | old | hw 25 + switch 0.4 / 1.5, dead_cost 3 / 8, hwd 12 | 0.6811 / 0.6823 / 0.6779 / 0.6834 / 0.6834 | | |
| old | team tree, base features, d12 | hw 0 / 25 | 0.6577 / 0.6745 | 0.5584 | 1812 |
| old | team tree + extra, d12 | hw 0 / 25 | 0.6812 / 0.6932 | 0.5813 | 1847 |
| old | team + extra, d6 / d8 / d10 / d14 | hw 25 | 0.6687 / 0.6835 / 0.6939 / 0.6960 | | |
| dead d4 / d6 / d8 / d10 / d12 | team + extra, d12 | hw 25 | 0.6861 / 0.6895 / 0.6937 / 0.6952 / 0.6958 | | |
| **dead d8** | **team + extra, d10** | **hw 25** | **0.6944** | **0.5815** | **1850** |
| dead d8 | team + extra d10 | hw 0 / 12 / 37 | 0.6802 / 0.6869 / 0.6924 | | |
| dead d8 | team + extra d10 | hw 25 + contrast 0 / switch 1.5 / switch 0.4 / hwd 12 / control 2.5 s | 0.6927 / 0.6952 / 0.6936 / 0.6969 / 0.6957 | | |
| dead d10 | team + extra d12 | hw 25 / 12 / 37 / contrast 0 / switch 1.5 | 0.6952 / 0.6900 / 0.7003 / 0.6963 / 0.6957 | | |

The shipped setting is dead depth 8 plus team depth 10. It is the smallest pair on the plateau
(1150 nodes in total, against 1885 for the old tree). Every other decoder constant was kept at
LDE's value, because no change moved the score beyond noise. The larger pairs and other decoder
values (up to 0.7003) are within about 0.006 of it and were picked from the same 21 game-seeds,
so they were not taken.

**Caveat.** The fold trees are leave-one-game-out. The hw = 25 smoothing and the two depths were
chosen on the same 21 game-seeds that are reported. Neighbouring values are close (hw 12-37
spans 0.687-0.694 on the shipped pair), so the optimism should be small, but it is not zero.

## Final scores (bench.py itself)

| bench | result |
|---|---|
| DFL `--dfl2` (fold trees) | **ctrl_mean 0.6945** (T=1/2/3: 0.649 / 0.720 / 0.714); raw spellF1 0.5815 ± 0.012; macroF1 0.9034; 1850 possessions per seed |
| Metrica dev | spellF1 0.6447 (decision_tree 0.6358) |
| match (`--match`, twice) | 325 possessions, 46.4% dead, 0 X; both runs byte-identical, sha256 `e4cef14a…cdaf` |

Paired against `decision_tree`'s saved bench-v2 codes (`probs/decision_tree`), same game and
seed:

| | decision_tree | decision_tree_a | difference |
|---|---|---|---|
| ctrl_mean | 0.6661 | 0.6944 | **+0.0283**, SE 0.0043, 20/21 game-seeds, 7/7 games (game-level SE 0.0058) |
| raw spellF1 | 0.5595 | 0.5815 | +0.0220, 21/21 |
| F1 H / A / D (pooled frames) | 0.875 / 0.861 / 0.967 | 0.881 / 0.870 / 0.967 | team +0.006 / +0.009, dead equal |
| team flips per alive minute (evidence argmax; truth 6.4) | 13.2 | 4.6 | |
| frame team accuracy, alive | 0.878 | 0.893 | |

Per game, the seed-mean difference is +0.025, +0.045, +0.048, +0.009, +0.034, +0.026 and +0.010
(games in `sorted(os.listdir(idsse/npz))` order). The only game-seed lost is J03WOY / 2027, at
-0.008.

References on the same bench: ensemble_ctrl 0.728, ensemble_vote 0.696, learned_dfl_evidence +
fold 0.698 (0.672 is without the fold; corrected 2026-11-08). decision_tree_a is not tested as an ensemble member here.

## What the trees say (top levels)

`python candidates/decision_tree_a/explain.py tree_dead.npz --depth 3`. The dead tree has depth
8, 225 nodes and uses 60 features:

```
if ball_disp_b <= 0.58 m or ball missing                                   [50%: D 0.86]
  if any team controlled the ball within the last 3.0 s                    [ 7%]
    ball out (ball_out over 2 s > 0)                    -> D               [1.6%: D 0.83]
    else                                                -> alive           [5.8%: alive 0.77]
  else
    players' mean speed 0.8 s later <= 2.24 m/s        -> D               [41%: D 0.97]
    faster                                              -> D (uncertain)   [2.4%: D 0.54]
else (ball moving)                                                         [50%: alive 0.97]
  ball within 0.24 m of a line: still-ish ball (disp <= 1.28 m) -> D, else alive
  else: mean speed over 5 s <= 0.95 m/s -> D, else alive                   [48%: alive 0.99]
```

`python candidates/decision_tree_a/explain.py tree_team.npz --depth 4`. The team tree has depth
10, 925 nodes and uses 119 features; it is trained on alive frames only:

```
if near-ball balance over the last 3 s favours away (nearbal_b3 < 0)        [47%: A 0.87]
  if away's nearest player was clearly closer over the last 3 s (margin_b3 <= -1.5 m)
                                                                           [35%: A 0.93]
    ... and home has not been near the ball for > 5.1 s       -> A         [23%: A 0.98]
  else (margin unclear)                                                    [12%: A 0.68]
    the balance over the last 1 s, the next 3 s and the ball's forward movement decide
else (balance favours home or nobody was near)                             [53%: H 0.83]
  mirror image: margin_b3 > 1.4 m, last control home, away not near for > 5.1 s -> H 0.98
  margin unclear and the last confident control was not home -> A 0.57 (the weak case)
```

In plain words: **the team whose players have been nearest the ball over the last 3 s has it.**
That rule alone is 87% right, and the closest-player margin over the same window raises it to
93-98%. Feature importance: nearbal_b3 0.71, margin_b3 0.10, ball_xdir_pmf 0.026, lastctrl_m0.5
0.023 and nearbal_b1 0.020; the rest are below 0.015 each. The weak part is still the moment
after a loose ball, when the window balance is near zero.

## Verdict

- It is a clear improvement on the teammate's tree: +0.028 ctrl_mean (SE 0.004), 20/21
  game-seeds, with a smaller tree (1150 nodes against 1885). About half of the gain comes from
  the decoder side (smoothing the team log-odds, +0.015). The other half comes from giving the
  tree window features it can split on (+0.020 over a team tree without them).
- It is level with learned_dfl_evidence + fold (0.698, boosted, same base features; corrected 2026-11-08), and it
  is level with ensemble_vote (0.696). It is still 0.034 below ensemble_ctrl (0.728). Short
  spells remain out of reach: the tree, like the boosters, mostly under-splits (1850
  possessions per seed against 3061).
- The smoothing and the near-balance features are model-agnostic. Before an ensemble decision,
  it is worth testing whether the same ±1 s team-logit smoothing helps LDE. That was not done
  here.

## Rebuild

```sh
PY=$MTAH_WORK/venv/bin/python; W=$MTAH_WORK/arch/decision_tree_a
# features at ds_v2's stride-5 rows (cached in $W/data) + all fold / full trees (about 22 min on 2 cores)
OMP_NUM_THREADS=1 $PY candidates/decision_tree_a/train.py --kind team --feats ext base --leaf 200 --depths 6 8 10 12 14 --jobs 2
OMP_NUM_THREADS=1 $PY candidates/decision_tree_a/train.py --kind dead --feats base --leaf 200 --depths 4 6 8 10 12 --jobs 2
# variant table (any row; "old" = decision_tree's fold trees, read-only)
OMP_NUM_THREADS=1 $PY candidates/decision_tree_a/sweep.py '{"dead":"old","team":"old","hw":25}' \
    '{"dead":"dead_base_L200_d8","team":"team_ext_L200_d10","hw":25}'
# install the chosen pair
cp $W/sweep/dead_base_L200_d8/tree_full.npz candidates/decision_tree_a/tree_dead.npz
cp $W/sweep/team_ext_L200_d10/tree_full.npz candidates/decision_tree_a/tree_team.npz
for g in 0 1 2 3 4 5 6; do cp $W/sweep/dead_base_L200_d8/fold_$g.npz $W/fold_dead_$g.npz
                           cp $W/sweep/team_ext_L200_d10/fold_$g.npz $W/fold_team_$g.npz; done
# $W/folds.json: {"<sha256(tree_dead.npz)[:12]>": ["$W", "dead"], "<sha256(tree_team.npz)[:12]>": ["$W", "team"]}
#   currently {"e59f74bc6b8a": [.., "dead"], "8c29de73968c": [.., "team"]}
OMP_NUM_THREADS=2 $PY candidates/bench.py candidates/decision_tree_a/pipeline.py --dfl2 --folds $W/folds.json \
    --save-probs $MTAH_WORK/probs/decision_tree_a
$PY candidates/bench.py candidates/decision_tree_a/pipeline.py --match     # dev score + submission.csv
$PY candidates/decision_tree_a/explain.py tree_team.npz --depth 4
```

Files: `pipeline.py` (candidate: `predict`, `predict_proba`; numpy only at inference),
`extra.py` (window features), `train.py`, `sweep.py`, `explain.py`, `tree_walk.py` and
`decoder.py` (copies from decision_tree, unchanged). Logs are in
`$MTAH_WORK/arch/decision_tree_a/` (`train_sweep1.log`, `bench_dfl2.log`, `bench_match{1,2}.log`).
