# nn_pff: the network members retrained on PFF World Cup 2022 + DFL

This folder holds the training code for networks trained on the 64 PFF FC World Cup 2022 games
(48 used, 16 held out) together with the 7 DFL games, plus the MS-TCN member's numpy inference.
Its sibling folders hold the other two members: `../nn_pff_tcn` (TCN member, numpy inference) and
`../ball_radar_pff` (Ball Radar + stacked LightGBM retrained the same way). The ensemble is
`../ensemble_ctrl_v6`. Results: the tables at the end and `../ensemble_ctrl_v6/README.md`.

## Data and target

- **PFF** (`$MTAH_WORK/pff_wc2022/npz`, `candidates/pff/`): 64 games in the IDSSE npz format, 25 fps,
  labels = the "touch" reading (team of the latest on-the-ball PFF event; length fingerprint
  0.977 / 0.902 / 0.632 at 314 possessions / 90, against the competition truth's published
  0.982 / 0.893 / 0.654 at about 306 / 90).
- **DFL** (`$MTAH_WORK/idsse/npz`): 7 games, raw BallStatus / BallPossession flags.
- **Split, fixed before any training** (`split_games.py`, numpy seed 20261108): 16 PFF games are the
  final holdout (`$MTAH_WORK/arch/nn_pff/holdout_games.json`), never trained, tuned or selected on.
  The other 48 form 4 game-grouped folds of 12 (`selection_folds.json`). Pre-registered protocol:
  `$MTAH_WORK/arch/nn_pff/prereg.txt`.
- **Attack direction:** the bench's kick-off rule (`tune_on_sample_data.attack_sign`).
  `check_signs.py` checks every game against both goalkeepers' mean x per half: 71/71 consistent.
  The old half-mean rule would reverse DFL J03WR9 and 6 PFF games (3813, 3815, 3819, 3821, 3822,
  3826). Log: `$MTAH_WORK/arch/nn_pff/check_signs.log`.
- **Degradation** (`make_dataset.py`; training seeds only, never the bench seeds 2027-2029):
  - `ds_pff2` (used by the "v5c+PFF" members): degrade_v2 with `make_dataset_v2.draw_params`
    (randomised knobs) + AR(1) ball offset, seeds 41 / 42 / 43, as `ds_v2s`. DFL = the `ds_v2s` files.
  - `ds_pff` (the architecture experiments): seeds 41 / 42 degrade_v2 as above, seed 44 =
    degrade_v4 (dead-ball islands, rolling exits, long open-play gaps) with knobs randomised
    around its defaults (`draw_params_v4`); DFL = ds_v2s + one degrade_v4 seed.
  - Features: learned_dfl_evidence's `features.compute` (194 identity-free columns), one stride-5
    phase per degradation (`idx % 5 == seed % 5`).

## Folds and enforcement

Every configuration trains 12 models (`data.py` specs): `pff<k>` (all training data except PFF
fold k) for the out-of-fold PFF scores; `dfl<g>` (all except DFL game g; file `fold_tcn_<g>.npz`,
the name `dfl_folds.resolve` looks up) for the DFL leave-one-game-out bench; `full` (48 PFF + 7
DFL) for the match, Metrica, the SKC check and the PFF holdout. Every file stores its
`train_games`.

- DFL: `bench.py --dfl2 --folds <folds_dfl.json>`, where the JSON is
  `arch/ensemble_ctrl_v5c/folds.json` plus the full model's sha256 mapped to the config folder.
- PFF: `run_pff.py` (selection mode) sets `NN_PFF_FOLD` / `NN_PFF_GAME` per game.
  `pff_folds.resolve` then swaps the weight file for `fold_pff_<k>` and refuses a model whose
  `train_games` contain the game. It also logs every lookup, and the runner stops if a game was not
  resolved to an out-of-fold model. DFL-only members (v5c's) never saw PFF and use their full
  models. `--holdout --final` runs the 16 holdout games with full models; it refuses to run
  twice.

## Models

- **MS-TCN** (`train_mstcn.py`; numpy `tcn_np.py` and decoder `pipeline.py` = handover_mstcn_v3's):
  ASRF backbone (C 64, 10 dilated layers) with a class head and a boundary head, 2 class
  refinement stages and 1 boundary refinement stage, CE + TMSE + boundary BCE, team-swap
  augmentation. "v5c+PFF" recipe (`m5c`): handover_mstcn_v3's settings (constant-lr Adam 5e-4, no
  clipping, dropout 0.5, crop 1024 x 8). Training is longer for the larger data: 40 epochs x 1200
  crops instead of 15 x 600.
- **TCN** (`train_tcn.py`; numpy + LDE decoder in `../nn_pff_tcn` = temporal_tcn_v3's): C 64,
  12 residual dilated blocks (receptive field 108 s at 5 fps), 3 class logits + hand-over logit,
  OneCycle AdamW 2e-3, crop 512 x 16. "v5c+PFF" recipe (`t5c`): temporal_tcn_v3's settings with
  4000 iterations instead of 800.
- **Ball Radar** (`../ball_radar_pff`): ball_radar_v3's set-attention + Bi-GRU radar
  (3200 iterations instead of 800), then its stacked LightGBM (dead / team / hand-over) on
  features of the radar-imputed ball, all retrained on the same fold specs. Radar inputs use the
  recipe's own degradation (degrade_dfl, seeds 31-33). The DFL inputs are bit-identical to
  ball_radar_v3's.
- Training ran on Apple MPS (float32, `--device mps`) after the first experiments ran on CPU. A
  few models were trained on CPU during a GPU pause (the `device` field in each file's
  `train_cfg`). Torch training is not bit-reproducible. The exported numpy weights are the
  artefacts, and `verify_export.py` checks that numpy, torch-CPU and torch-MPS outputs of an
  exported model agree (max |diff| ~2e-7 on probabilities).

Inference is numpy only (plus LightGBM for the radar stack) and deterministic.


## Results

All tables come from `report.py` (saved codes; DFL = bench --dfl2 leave-one-game-out codes,
7 games x 3 seeds, fine4; PFF = out-of-fold codes on the 48 selection games, fine3 = mean of touch
spellF1 and its 0.5 s / 1 s folds; paired = mean difference, SE over games, game(-seed) wins).
The 16 PFF holdout games are not in any table here. They were scored once, by the coordinator,
for the frozen v6 (`../ensemble_ctrl_v6/README.md`, "Final pick").

**PFF caveat.** PFF's ball is placed on the carrier at event times, and its touch labels come from the
same events. So PFF-trained models gain far more on PFF (+0.10) than on DFL (+0.01 to +0.03).
DFL, SkillCorner and Metrica are the transfer tests. Members are always scored alone, with their own
decoder.

### Ensembles (paired vs ensemble_ctrl_v5c)
| | DFL fine4 | vs ref | PFF fine3 | vs ref | PFF touch | Metrica dev |
|---|---|---|---|---|---|---|
| ensemble_ctrl_v5c (reference) | 0.7171 | +0.0000 ± 0.0000 (0/21) | 0.6817 | +0.0000 ± 0.0000 (0/48) | 0.6635 | 0.6974 |
| ensemble_ctrl_v4 | 0.7095 | -0.0076 ± 0.0038 (4/21) | 0.6761 | -0.0056 ± 0.0025 (16/48) | 0.6580 | 0.7083 |
| v6 members, 0.75 s rule (pipeline_v5cpff.py, live) | 0.7327 | +0.0156 ± 0.0060 (15/21) | 0.7887 | +0.1070 ± 0.0036 (48/48) | 0.7718 | 0.7298 |
| v6 frozen = members + v5c rule (pipeline.py; DFL live, PFF offline vote*) | 0.7286 | +0.0115 ± 0.0040 (16/21) | 0.7815* | +0.0998 ± 0.0032 (48/48)* | 0.7616* | 0.7285 |

\* the offline vote of the members' out-of-fold codes; the offline vote equals the live pipeline on
DFL (21/21) and SkillCorner (20/20). Metrica dev from the coordinator's live run.

### Members alone (paired vs ensemble_ctrl_v5c)
| | DFL fine4 | vs ref | PFF fine3 | vs ref | PFF touch | Metrica dev |
|---|---|---|---|---|---|---|
| temporal_tcn_v3 | 0.7028 | -0.0143 ± 0.0045 (2/21) | 0.6549 | -0.0268 ± 0.0027 (3/48) | 0.6412 | 0.6653 |
| handover_mstcn_v3 | 0.7080 | -0.0091 ± 0.0048 (7/21) | 0.6637 | -0.0181 ± 0.0019 (3/48) | 0.6506 | 0.6544 |
| ball_radar_v3 | 0.6781 | -0.0390 ± 0.0054 (1/21) | 0.6538 | -0.0279 ± 0.0028 (2/48) | 0.6367 | 0.7038 |
| t5c (TCN, v5c+PFF) | 0.7300 | +0.0129 ± 0.0057 (15/21) | 0.7809 | +0.0991 ± 0.0040 (48/48) | 0.7648 | 0.7142 |
| m5c (MS-TCN, v5c+PFF) | 0.7251 | +0.0080 ± 0.0087 (11/21) | 0.7868 | +0.1051 ± 0.0035 (48/48) | 0.7728 | 0.6979 |
| radar_pff (v5c+PFF) | 0.6988 | -0.0183 ± 0.0061 (4/21) | 0.7215 | +0.0398 ± 0.0033 (46/48) | 0.7046 | 0.7345 |

### Width x context (paired vs the grid base of the same family): TCN
| | DFL fine4 | vs ref | PFF fine3 | vs ref | PFF touch | Metrica dev |
|---|---|---|---|---|---|---|
| TCN base C64 / RF 108 s `e3_tcn` | 0.7224 | +0.0000 ± 0.0000 (0/21) | 0.7759 | +0.0000 ± 0.0000 (0/48) | 0.7599 | - |
| TCN width C128 `gW_tcn` | 0.7347 | +0.0123 ± 0.0029 (19/21) | 0.7750 | -0.0008 ± 0.0019 (23/48) | 0.7593 | - |
| TCN context RF 415 s, crop 2048 `gX_tcn` | 0.7224 | -0.0000 ± 0.0037 (8/21) | 0.7703 | -0.0056 ± 0.0018 (19/48) | 0.7542 | - |
| TCN width + context `gWX_tcn` | 0.7263 | +0.0039 ± 0.0040 (12/21) | 0.7728 | -0.0031 ± 0.0020 (17/48) | 0.7575 | - |

### Width x context (paired vs the grid base of the same family): MS-TCN
| | DFL fine4 | vs ref | PFF fine3 | vs ref | PFF touch | Metrica dev |
|---|---|---|---|---|---|---|
| MS-TCN base C64 L10 crop 1024 `e1_base` | 0.7123 | +0.0000 ± 0.0000 (0/21) | 0.7637 | +0.0000 ± 0.0000 (0/48) | 0.7496 | - |
| MS-TCN width C128 `gW_mstcn` | 0.7298 | +0.0175 ± 0.0028 (20/21) | 0.7927 | +0.0290 ± 0.0022 (46/48) | 0.7790 | - |
| MS-TCN context L12 crop 4096 `gX_mstcn` | 0.7280 | +0.0157 ± 0.0036 (17/21) | 0.7871 | +0.0234 ± 0.0022 (43/48) | 0.7737 | - |
| MS-TCN width + context `gWX_mstcn` | 0.7344 | +0.0220 ± 0.0045 (19/21) | 0.7966 | +0.0329 ± 0.0027 (46/48) | 0.7834 | - |

### Optimisation / loss / data axes on the TCN recipe (paired vs t5c)
| | DFL fine4 | vs ref | PFF fine3 | vs ref | PFF touch | Metrica dev |
|---|---|---|---|---|---|---|
| base (t5c) `t5c` | 0.7300 | +0.0000 ± 0.0000 (0/21) | 0.7809 | +0.0000 ± 0.0000 (0/48) | 0.7648 | - |
| C128 (width) `t5c_W` | 0.7328 | +0.0028 ± 0.0065 (13/21) | 0.7781 | -0.0028 ± 0.0020 (19/48) | 0.7625 | - |
| seed 1 `o_seed1` | 0.7290 | -0.0010 ± 0.0022 (9/21) | 0.7782 | -0.0026 ± 0.0017 (19/48) | 0.7623 | - |
| seed 2 `o_seed2` | 0.7268 | -0.0032 ± 0.0035 (10/21) | 0.7808 | -0.0001 ± 0.0019 (23/48) | 0.7641 | - |
| 3-seed softmax average (t5c + seeds 1, 2) `seeds3` | 0.7290 | -0.0010 ± 0.0028 (11/21) | 0.7849 | +0.0041 ± 0.0013 (32/48) | 0.7682 | - |
| EMA 0.999 `o_ema` | 0.7271 | -0.0029 ± 0.0021 (8/21) | 0.7786 | -0.0023 ± 0.0011 (22/48) | 0.7621 | - |
| focal gamma 2 `o_focal2` | 0.7230 | -0.0070 ± 0.0059 (7/21) | 0.7825 | +0.0016 ± 0.0023 (28/48) | 0.7630 | - |
| boundary-weighted CE (trans_w 1) `o_transw1` | 0.7289 | -0.0011 ± 0.0025 (9/21) | 0.7806 | -0.0002 ± 0.0015 (26/48) | 0.7649 | - |
| hand-over loss weight 1.0 `o_cw1` | 0.7321 | +0.0021 ± 0.0015 (14/21) | 0.7831 | +0.0023 ± 0.0016 (28/48) | 0.7671 | - |
| label smoothing 0.05 `o_ls05` | 0.7384 | +0.0084 ± 0.0025 (19/21) | 0.7947 | +0.0139 ± 0.0018 (43/48) | 0.7768 | - |
| lr x0.5 `o_lr05` | 0.7209 | -0.0091 ± 0.0018 (2/21) | 0.7690 | -0.0119 ± 0.0020 (7/48) | 0.7527 | - |
| lr x2 `o_lr2` | 0.7278 | -0.0022 ± 0.0023 (6/21) | 0.7831 | +0.0022 ± 0.0014 (29/48) | 0.7672 | - |
| cosine schedule `o_cos` | 0.7271 | -0.0029 ± 0.0024 (7/21) | 0.7807 | -0.0002 ± 0.0016 (22/48) | 0.7647 | - |
| weight decay 1e-3 `o_wd1e3` | 0.7307 | +0.0007 ± 0.0013 (14/21) | 0.7792 | -0.0016 ± 0.0010 (17/48) | 0.7632 | - |
| DFL sampling share 0.30 `o_wdfl30` | 0.7308 | +0.0009 ± 0.0050 (14/21) | 0.7761 | -0.0048 ± 0.0018 (19/48) | 0.7603 | - |
| fine-tune on DFL after joint `o_ftdfl` | 0.7292 | -0.0008 ± 0.0038 (10/21) | 0.7570 | -0.0238 ± 0.0023 (3/48) | 0.7409 | - |
| + SkillCorner 15 games `t5c_skc` | 0.7261 | -0.0039 ± 0.0034 (7/21) | 0.7722 | -0.0086 ± 0.0015 (11/48) | 0.7556 | - |
| + FOOTPASS 38 games `t5c_fp` | 0.7277 | -0.0023 ± 0.0024 (10/21) | 0.7737 | -0.0071 ± 0.0019 (14/48) | 0.7574 | - |

### Variants of the MS-TCN recipe (paired vs m5c)
| | DFL fine4 | vs ref | PFF fine3 | vs ref | PFF touch | Metrica dev |
|---|---|---|---|---|---|---|
| base (m5c) `m5c` | 0.7251 | +0.0000 ± 0.0000 (0/21) | 0.7868 | +0.0000 ± 0.0000 (0/48) | 0.7728 | - |
| C128 / Cr 64 (width) `m5c_W` | 0.7324 | +0.0073 ± 0.0047 (15/21) | 0.7989 | +0.0122 ± 0.0018 (42/48) | 0.7854 | - |
| label smoothing 0.05 `m5c_ls` | 0.7264 | +0.0013 ± 0.0032 (12/21) | 0.7903 | +0.0035 ± 0.0015 (32/48) | 0.7765 | - |
| + SkillCorner 15 games `m5c_skc` | 0.7250 | -0.0001 ± 0.0018 (10/21) | 0.7775 | -0.0093 ± 0.0024 (12/48) | 0.7632 | - |

Notes on the axes:
- **Width x context.** TCN: width helps on DFL only on the ds_pff base (+0.012); on the best recipe
  (t5c, ds_pff2) it gives nothing (t5c_W +0.003 ± 0.007). Context hurts the TCN on PFF (-0.006). MS-TCN:
  width helps on both benches (base +0.0175 DFL / +0.029 PFF; on the m5c recipe +0.007 / +0.012).
  Context also helps on its base (+0.016 / +0.023), and width + context is best (+0.022 / +0.033).
  The crop-4096 runs are 10-20x slower to train, and context was not tried on the m5c recipe:
  this is the most promising untested direction. (gWX_mstcn's `full` model was not trained; its
  DFL bench used `fold_tcn_0.npz` as the sha anchor, which still maps every game to its own fold.)
- **Seeds / EMA.** Single-seed spread on DFL is about 0.003. A 3-seed softmax average gives PFF +0.004
  and DFL -0.001. EMA 0.999 is slightly worse. Neither is worth its cost.
- **Loss.** Label smoothing 0.05 is the only clear gain for the TCN (DFL +0.0084 ± 0.0025, 19/21;
  PFF +0.014). For the MS-TCN it is +0.001 / +0.004. Focal loss (gamma 2) coarsens the output (sel2
  +0.014, fine4 -0.007). Boundary-weighted CE and the hand-over weight are within noise.
- **Optimiser.** lr x0.5 is clearly worse. lr x2, cosine instead of OneCycle, and weight decay
  1e-3 are within noise: the recipe's settings are at or near the optimum.
- **Data mix / fine-tuning.** Raising DFL's sampling share to 0.30 and fine-tuning on DFL after
  joint training both lose PFF (-0.005 / -0.024) without gaining DFL. Joint training at the natural
  share stays.
- **Extra data.** +SkillCorner (15 games) and +FOOTPASS (38 games, synthetic ball) both lower DFL
  and PFF. +SkillCorner raises its own 5-game holdout (+0.049 TCN, +0.055 MS-TCN), so it is an
  in-domain gain only. Neither is kept.
- **Degradation.** The degrade_v2-only data (ds_pff2: t5c, m5c) beat the v2 + v4 mix (ds_pff: e3, e1)
  on both benches: TCN DFL 0.7300 vs 0.7224, MS-TCN 0.7251 vs 0.7123 (MS-TCN also differs in its
  schedule). Both benches are degrade_v2, so part of this may be bench matching.

### Candidates for the user (post-hoc, not frozen)

| vote with v5c's rule, offline (members' LOGO / out-of-fold codes), paired vs frozen v6 | DFL fine4 | PFF fine3 | SkillCorner fine3 | Metrica |
|---|---|---|---|---|
| m5c -> m5c_W | +0.0057 ± 0.0022 (17/21) | +0.0076 ± 0.0011 (40/48) | +0.0012 ± 0.0018 (9/20) | +0.001 |
| t5c -> o_ls05 | -0.0000 ± 0.0024 (11/21) | +0.0017 ± 0.0010 | +0.0028 ± 0.0016 | +0.006 |
| m5c -> m5c_ls | +0.0021 ± 0.0013 (14/21) | +0.0024 ± 0.0009 | +0.0043 ± 0.0019 | +0.006 |
| t5c -> o_ls05, m5c -> m5c_ls | +0.0027 ± 0.0014 (13/21) | +0.0035 ± 0.0010 | +0.0058 ± 0.0027 | +0.008 |
| **t5c -> o_ls05, m5c -> m5c_W (`pipeline_lsw.py`, also live)** | **+0.0099 ± 0.0021 (17/21)** | **+0.0076 ± 0.0016 (37/48)** | **+0.0054 ± 0.0026 (13/20)** | +0.0035 |

Logs: `$MTAH_WORK/arch/nn_pff/logs/swap_*.txt` (`swap_eval.py`).

## Reproduce (main steps)

```
W=$MTAH_WORK; P=$W/venv/bin/python
$P -I candidates/nn_pff/split_games.py ; $P -I candidates/nn_pff/check_signs.py
$P -I candidates/nn_pff/make_dataset.py                                  # ds_pff
$P -I candidates/nn_pff/make_dataset.py --out $W/ds_pff2 --pff-seeds 41 42 43 --dfl-new-seeds --link-from $W/ds_pff
SPECS=pff0,pff1,pff2,pff3,dfl0,dfl1,dfl2,dfl3,dfl4,dfl5,dfl6,full
$P -I candidates/nn_pff/train_tcn.py   --device mps --ds $W/ds_pff2/ds --out $W/arch/nn_pff/t5c --specs $SPECS
$P -I candidates/nn_pff/train_mstcn.py --device mps --ds $W/ds_pff2/ds --out $W/arch/nn_pff/m5c --specs $SPECS --cfg '{"cosine":0,"clip":0}'
# every other config: $W/arch/nn_pff/<config>/train_cfg.json; the job list is $W/arch/nn_pff/jobs_gpu.txt
candidates/nn_pff/chain.sh <config> tcn|mstcn       # PFF out-of-fold (run_pff.py) + bench --dfl2
python candidates/nn_pff/report.py                  # the tables above
```

