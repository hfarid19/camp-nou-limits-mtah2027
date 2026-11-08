# ensemble_ctrl_v5c: v4's vote with all three members retrained on corrected training data

`pipeline.py` is `ensemble_ctrl_v4`'s pipeline with a different member list:
a hard majority vote of `temporal_tcn_v3`, `handover_mstcn_v3` and `ball_radar_v3` (that tie-break
order), then `merge_short_v2` at 1.0 s, `control.fold` at 1.25 s and `prekick_dead`, all unchanged.
`members.py` and `ensemble_core.py` are copies of v4's.

## The finding: one training game had its attack direction reversed

Every member's training set takes the attack direction (`sign`: does the home team attack +x in the
first half?) from `learned_dfl_evidence/make_dataset.attack_sign`. That function compares the home
team's mean x over the whole first half with the halfway line. The bench (`tune_on_sample_data.attack_sign`)
looks at the first 50 frames of the first half instead, and the match takes the direction from its metadata.

| game | home x at the kick-off | home mean x over the first half | bench sign | training sign |
|---|---|---|---|---|
| DFL-MAT-J03WR9 | 45.7 m | 54.9 m | **+1** | **-1** |
| the other 6 games | 55-65 m | 52.5-63 m | -1 | -1 |

So for 1 of the 7 games (3 of ds_v2's 21 degradations, and the matching ball_radar
sequences and stacked features), every direction-dependent feature was mirrored relative to the
labels. The bug was found when I ran the members' fold models on the training degradations. The
held-out J03WR9 model scored a spellF1 of 0.25 there, against about 0.65 elsewhere, and it reached
0.71 on the bench copies of the same game.

**Fix.** `candidates/fix_attack_sign_ds.py` builds `$MTAH_WORK/ds_v2s`. Unchanged files are
hard-linked. J03WR9 is re-signed and its features recomputed with the same code (the recompute
reproduces a stored ds_v2 file bit-for-bit). The members were then retrained with their unchanged
code and recipes: `temporal_tcn_v3`, `handover_mstcn_v3` (raw DFL-flag target, seed 0) and
`ball_radar_v3` (its `prep_data.py` / `feat_data.py` use the kick-off rule). See each folder's README.

## Members (bench --dfl2, live, LOGO, paired vs the v4 member; SE over 7 game means)

| member | raw spellF1 | fine4 | fine4 difference | J03WR9 (control) |
|---|---|---|---|---|
| temporal_tcn_v3 | 0.6641 -> **0.6737** | 0.6931 -> **0.7028** | +0.0096 ± 0.0040 (15/21) | 0.0000 |
| handover_mstcn_v3 | 0.6703 -> **0.6823** | 0.6961 -> **0.7080** | +0.0119 ± 0.0029 (16/21) | 0.0000 |
| ball_radar_v3 | 0.6311 -> **0.6445** | 0.6643 -> **0.6781** | +0.0139 ± 0.0037 (17/21) | -0.0005 |

The J03WR9 fold models of the two TCNs train on exactly the same data as before (J03WR9 is held
out), and torch reproduces them bit-for-bit. That game therefore checks for retraining noise, and
the gains on the other six games come from the fix. The radar was retrained with 2 threads instead
of 1, so its J03WR9 fold also differs slightly (-0.0005).

## Ensemble (selection metric fine4; nested = choice made on 6 DFL games, scored on the 7th)

| vs ensemble_ctrl_v4 | fine4 | sel2 | raw spellF1 | macroF1 |
|---|---|---|---|---|
| **v5c, v4's rule (live, dfl2)** | **+0.0076 ± 0.0038 (17/21)** | +0.0053 ± 0.0037 (15/21) | +0.0071 ± 0.0036 (18/21) | +0.0024 ± 0.0008 (18/21) |
| v5c members, smoothing x fold re-chosen (nested) | +0.0062 ± 0.0046 (16/21) | +0.0028 ± 0.0049 | +0.0061 ± 0.0042 | +0.0024 ± 0.0007 |
| v5c on bench dfl3 (degrade_v3, live) | +0.0063 ± 0.0039 (15/21) | +0.0029 ± 0.0035 | | |
| **Metrica dev (Sample Games 1+2, 2 seeds)** | spellF1 **0.6974 vs 0.7084 (-0.011, 0/4 game-seeds)** | | | macroF1 0.8246 vs 0.8250 |

The nested re-selection picks v4's own rule (1 s smoothing, 1.25 s fold) in 6 of 7 folds, so the
rule is kept. Per game, the live fine4 differences are +0.013 +0.015 +0.022 -0.005 +0.010 -0.004 +0.002.
Absolute live scores (dfl2) are fine4 0.7171, sel2 0.7463, raw 0.6808, macroF1 0.9296 and
ctrl_mean 0.7236.

**The Metrica cross-check goes the other way.** Across all 8 old/new member mixes, every swap to a
v3 member raises DFL fine4 and lowers Metrica dev: from v4 at 0.7095 / 0.7083 to v5c at
0.7171 / 0.6974 (`_scratch/v5c_dev_combos.py`). At member level on Metrica, the TCN gains +0.019
while the MS-TCN loses -0.003 and the radar -0.005. The Metrica attack signs were checked and are
correct. Metrica dev has 2 games and uses the old v1 degradation, which the repo already found
unlike the competition tracking. The two DFL benches (dfl2 and dfl3) both favour v5c.

## Ideas tried without gain (offline on saved LOGO codes and probabilities, nested on fine4 vs v4)

| idea | nested fine4 vs v4 |
|---|---|
| 5-member votes: v4's 3 + any 2 of tcn_ctrl, hm_ctrl, tcn_ctrl_mix, tcn_v1, hm_v1, lde, pitch_control, ballfree, v2_stacked (36 sets x smoothing x fold) | -0.0011 ± 0.0011 (picks v4 every time) |
| soft vote: pooled probabilities (arithmetic / geometric, 3 weightings) -> 3-state Viterbi -> merge -> fold | -0.0064 ± 0.0018 (sel2 +0.003, macroF1 +0.0015) |
| hard vote with the TCN's decoder replaced by a plain Viterbi of its own probabilities | +0.0006 ± 0.0017 (macroF1 +0.0025) |

The members' decoders matter more than the pooling. A plain Viterbi beats the TCN's own decoder
(0.701 vs 0.693), but it loses to the MS-TCN's and the radar's decoders, and the pooled soft vote
loses 0.006. A LOGO stacked LightGBM combiner trained on member outputs for the training seeds
was drafted (`_scratch/v5c_stack_*.py`) and dropped after these results; it was not run.
Mirroring is already in both TCNs' training (p = 0.5 crop swap) and inference.

## Match (positions only, no labels)

`submission.csv` has sha256 `5b85f374…`. Two runs are `cmp`-identical, and
`validate_submission.py` passes. The output has 329 possessions (v4: 322), dead share 0.48 and no
X inside the halves. 2.4% of graded frames differ from v4, 1.5% of them H<->A.

## Reproduce

```
W=$MTAH_WORK
python -I candidates/fix_attack_sign_ds.py                      # ds_v2s; then the members' READMEs
OMP_NUM_THREADS=2 python candidates/bench.py candidates/ensemble_ctrl_v5c/pipeline.py --dfl2 \
    --folds $W/arch/ensemble_ctrl_v5c/folds.json --save-probs $W/probs/ensemble_ctrl_v5c
python candidates/score_sel2.py $W/probs/ensemble_ctrl_v5c --ref $W/probs/ensemble_ctrl_v4
python candidates/bench.py candidates/ensemble_ctrl_v5c/pipeline.py --match
```

`$MTAH_WORK/arch/ensemble_ctrl_v5c/folds.json` is `ensemble_ctrl_v2/folds.json` plus
`90b2fe8df174` (temporal_tcn_v3), `459018cb80b4` (handover_mstcn_v3), and `6b161591d2b1` /
`9030306222e0` / `e0e3a8b5b683` / `003229528d22` (ball_radar_v3 radar / brd / brha / brc). The live
dfl2 codes equal the offline vote of the saved member codes on 21/21 game-seeds. Offline scripts:
`candidates/_scratch/v5c_*.py` (harness `v5c_common.py`, experiments `v5c_e2/e3/e3b/e4/e6.py`).
