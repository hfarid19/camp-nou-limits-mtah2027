# decision_tree

How far does one plain, interpretable decision tree get, and what does it learn?

## Method

- **Evidence:** one scikit-learn `DecisionTreeClassifier` (gini, best splits, `random_state=0`).
  It predicts the per-frame label H / A / D from learned_dfl_evidence's 194 identity-free features
  (`learned_dfl_evidence/features.py`).
- **Training data:** `$MTAH_WORK/ds_v2/ds/<match>_<seed>.npz`, made by `make_dataset_v2.py`: the 7
  DFL games under the randomised degrade_v2, seeds 41-43, stride 5, about 610k rows. Every row is
  also added team-swapped (`features.swap_index`, with the H and A labels swapped).
- **H/A symmetry:** at inference the tree's prediction on the swapped features is averaged in, so
  H/A is exactly symmetric.
- **Inference in plain numpy:** `train.py` exports the tree to `tree.npz` (feature, threshold,
  children, NaN direction, Laplace-smoothed leaf class frequencies), and `tree_walk.py` walks it in
  plain numpy. There is no scikit-learn or torch at inference, and the output is deterministic.
  `train.py` checks the walker against `predict_proba`: max |diff| 0 on 24k rows, two thirds of
  them with a NaN feature.
- **Decoder:** LDE's round-3 decoder, copied verbatim into `decoder.py`. It takes
  p_dead = P(D) and p_home = P(H) / (P(H) + P(A)). There is no hand-over model, so H<->A boundaries
  use the first-hit snap.
- **Control rule:** `pipeline.py` then applies `control.fold(codes, 2.0, runs)`, as
  `ensemble_ctrl` does. `pipeline_raw.py` is the same pipeline without the fold.
- **Folds:** `tree.npz` is trained on all 7 DFL games and is loaded through
  `dfl_folds.resolve`. The bench swaps in the fold files in
  `$MTAH_WORK/arch/decision_tree/folds/fold_tree_<gi>.npz`, each trained without game gi; the
  mapping is in `folds.json`, sha256 prefix `642af161de9b`.

## Selection (DFL leave-one-game-out only)

`sweep.py` scores every setting with the bench's own code: the same `dfl2_prepared` games and
seeds 2027-2029, `pm.evaluate` and `controlled_labels`, with fold trees. Each game's features
are computed only once.

Deeper trees are the top levels of one tree grown to the largest depth. That is identical to
growing with `max_depth`. Possessions are per seed, summed over the 7 games (truth: 3061).

| min_samples_leaf | depth | nodes (full) | ctrl_mean (fold 2 s) | ctrl_mean (raw decode) | spellF1 raw decode | macroF1 | poss (fold) |
|---|---|---|---|---|---|---|---|
| 200 / 2000 | 3 | 15 | 0.566 | 0.547 | 0.495 | 0.866 | 1856 |
| 200 | 5 | 63 | 0.597 | 0.564 | 0.515 | 0.877 | 1820 |
| 2000 | 5 | 61 | 0.597 | 0.565 | 0.515 | 0.877 | 1819 |
| 200 | 8 | 407 | 0.642 | 0.600 | 0.557 | 0.892 | 1858 |
| 2000 | 8 | 271 | 0.640 | 0.595 | 0.550 | 0.891 | 1846 |
| **200** | **12** | **1885** | **0.666** | 0.611 | 0.572 | 0.898 | 1876 |
| 2000 | 12 | 477 | 0.646 | 0.592 | 0.550 | 0.893 | 1865 |
| 100 | 12 | 2329 | 0.668 | 0.614 | 0.574 | 0.899 | 1894 |
| 100 | 16 | 4853 | 0.665 | 0.594 | 0.566 | 0.899 | 1930 |
| 100 | 20 | 5899 | 0.663 | 0.582 | 0.560 | 0.899 | 1964 |

The plateau is at depth 12. The chosen setting is **depth 12, min_samples_leaf 200**. L100_d12 is
+0.002 better, which is below the paired noise (about 0.003), and its tree is about a quarter
larger, so the smaller tree was kept. Deeper trees only fragment the output: more possessions,
and lower raw-decode scores.

## Final scores

The scores below are from bench.py itself.

| bench | ctrl_mean (T=1,2,3) | spellF1 (raw labels) | macroF1 | possessions |
|---|---|---|---|---|
| DFL `--dfl2`, `pipeline.py` (fold 2 s) | **0.6661** (0.624 / 0.690 / 0.684) | 0.5595 ± 0.013 | 0.8984 | 1876 / 3061 per seed |
| DFL `--dfl2`, `pipeline_raw.py` | 0.6108 (0.620 / 0.625 / 0.587) | 0.5724 ± 0.008 | 0.8990 | 2395 / 3061 |
| Metrica dev, `pipeline.py` | - | 0.6358 | 0.8151 | - |
| match (`--match`) | - | - | - | 342 possessions, 45.7% dead, 0 X |

Two `--match` runs are byte-identical: `submission.csv` sha256 `4950a272…`.

References on the same bench:

| candidate | ctrl_mean | raw spellF1 |
|---|---|---|
| ensemble_ctrl | 0.728 | - |
| ensemble_vote | 0.696 | 0.656 |
| learned_dfl_evidence | 0.698 (+fold); 0.672 without | 0.616 |
| decision_tree (fold) | 0.666 | 0.560 |
| decision_tree (raw) | 0.611 | 0.572 |

Against LDE, which uses the same features and decoder with boosted models, on bench-v2 codes:

| | F1 H | F1 A | F1 D | possessions per seed | recall of true spells <1 / 1-3 / 3-8 / >8 s |
|---|---|---|---|---|---|
| LDE raw | 0.885 | 0.874 | 0.968 | 2159 | 0.27 / 0.55 / 0.81 / 0.98 |
| tree raw | 0.872 | 0.859 | 0.967 | 2395 | 0.27 / 0.54 / 0.78 / 0.97 |
| LDE + fold 2 s | 0.880 | 0.869 | 0.968 | 1833 | 0.24 / 0.48 / 0.81 / 0.98 |
| tree + fold 2 s | 0.871 | 0.858 | 0.967 | 1876 | 0.23 / 0.47 / 0.78 / 0.97 |

(Recall here means that at least half of the true spell's frames have its label.)

**Dead ball is solved as well as LDE solves it.** The whole gap is the team call:
- H and A F1 are 0.013-0.015 lower.
- The raw output has about 240 more possessions per seed, because piecewise-constant leaf
  probabilities flip the team on single splits.
- Long spells (3-8 s) are recalled less often.

The control fold removes most of the flicker, which is why the tree gains +0.055 ctrl_mean from it
(LDE gains +0.026; corrected 2026-11-08, the earlier +0.056 compared against LDE without the fold).

## What the tree learned (top 4 levels of the shipped tree)

Run `python candidates/decision_tree/explain.py --depth 4`. Each node gives its share of training
rows and its class mix H / A / D. `ball_disp_b` is the ball displacement over the last 0.5 s, NaN
when the ball is missing. `lastctrl` is +1 / -1 / 0: the team of the last "confident control"
(a player within 1 m of the ball, nearest opponent beyond 1.5 m, for 3 frames). `*_since_ctrl` is
log1p(frames).

```
if ball_disp_b <= 0.58 m or ball missing                              [50%: D 0.86]
  if any team controlled the ball within the last 3.0 s               [ 7%]
    ball out of play (ball_out over 2 s > 0)          -> D            [ 2%: D 0.83]
    else                                              -> lastctrl team [ 6%: 0.69]
  else (no control for > 3.0 s)                                       [43%: D 0.95]
    players' mean speed 0.8 s later <= 2.24 m/s and ball speed <= 7.9 -> D  [41%: D 0.98]
    players moving faster                             -> D (uncertain, H/A 0.23 each, D 0.54)
else (ball moving)                                                    [50%: H 0.49 A 0.49]
  lastctrl = away                                                     [25%: A 0.82]
    away controlled within the last 3.4 s                             [21%: A 0.88]
      away's nearest player closer to the ball by > 1.25 m (2 s mean) -> A  [14%: A 0.94]
      otherwise                                       -> A            [ 7%: A 0.76]
    away control older than 3.4 s
      ball barely moving (disp <= 1.02 m)             -> D            [ 0.6%: D 0.78]
      else                                            -> A (weak: A 0.58)
  lastctrl = home: the mirror image -> H
```

In plain words:
- **Dead ball:** "the ball is missing or nearly still, nobody has had it under control for more
  than 3 s, and the players are walking (mean speed under about 2.2 m/s)" covers 41% of all frames
  at 98% purity. A ball that is out of the pitch is dead, even right after a touch.
- **Possession:** "while the ball moves, the team that last had a player alone on the ball keeps
  it". This is 82% right, and 94% right when that control was within the last 3.4 s and that
  team's nearest player is clearly closer to the ball than the other team's.
- **The weak part:** the deeper levels (about 140 of the 194 features are used somewhere)
  mostly patch the case where the last control is old, using team shape and the forward
  movement of the ball and the teams.

Feature importance (weighted gini decrease, whole depth-12 tree): ball_disp_b 0.51,
lastctrl 0.23, any_since_ctrl 0.07, ball_out_m2.0 0.016, H/A_since_ctrl 0.015 each,
diff_spread_m5.0 0.013, ball_xdir_pmf 0.012, mean_speed_s-0.8 0.012, diff_d1_m2.0 0.012.
Three features carry 80% of the gini decrease.

## Verdict

- **Not competitive alone.** ctrl_mean 0.666 is 0.032 below LDE with the same fold (0.698; corrected 2026-11-08, the earlier 0.672 was LDE without the fold),
  0.03 below ensemble_vote and 0.06 below ensemble_ctrl. Raw spellF1 is 0.04 below LDE.
- **Not a useful ensemble member.** As the third voter with temporal_tcn and handover_mstcn
  (plain frame majority vote, ties to temporal_tcn, then fold 2 s), the tree gives ctrl_mean
  0.7275, against 0.7252 with LDE in that slot and 0.7256 with both. That is
  interchangeable within noise. The tree disagrees with LDE on only 5% of graded frames, so it
  adds no new information.
- **Useful as an explanation.** Most of what the boosted LDE models gain comes from three
  features and two rules (a still or missing ball with no recent control means dead; otherwise
  the last controlling team has it). The rest of the ensemble's gain is in splitting short
  possessions, which a tree cannot do from per-frame features.

## Rebuild

```sh
PY=$MTAH_WORK/venv/bin/python        # scikit-learn 1.9.1 (requirements.txt), training only
OMP_NUM_THREADS=2 $PY candidates/decision_tree/train.py --leaf 200 2000 --depths 3 5 8 12 --jobs 2   # ~16 min
OMP_NUM_THREADS=2 $PY candidates/decision_tree/train.py --leaf 100 --depths 12 16 20 --jobs 2
OMP_NUM_THREADS=2 $PY candidates/decision_tree/sweep.py --jobs 2      # the selection table
W=$MTAH_WORK/arch/decision_tree
cp $W/sweep/L200_d12/tree_full.npz candidates/decision_tree/tree.npz
for g in 0 1 2 3 4 5 6; do cp $W/sweep/L200_d12/fold_tree_$g.npz $W/folds/fold_tree_$g.npz; done
# folds.json: {"<sha256(tree.npz)[:12]>": ["$W/folds", "tree"]}
$PY candidates/bench.py candidates/decision_tree/pipeline.py --dfl2 --folds $W/folds.json \
    --save-probs $MTAH_WORK/probs/decision_tree
$PY candidates/bench.py candidates/decision_tree/pipeline_raw.py --dfl2 --folds $W/folds.json \
    --save-probs $MTAH_WORK/probs/decision_tree/raw
$PY candidates/bench.py candidates/decision_tree/pipeline.py --match   # dev score + submission.csv
$PY candidates/decision_tree/explain.py --depth 4                      # the rules above
```

Logs: `$MTAH_WORK/arch/decision_tree/{train_sweep,sweep,sweep_deep,bench_dfl2,bench_dfl2_raw,bench_match1,bench_match2}.log`.
