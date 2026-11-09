# nn_pff: the network members retrained on PFF World Cup 2022 + DFL

This folder holds the training code for networks trained on the 64 PFF FC World Cup 2022 games
(48 used, 16 held out) together with the 7 DFL games, plus the MS-TCN member's numpy inference.
Its sibling folders hold the other two members: `../nn_pff_tcn` (TCN member, numpy inference) and
`../ball_radar_pff` (Ball Radar + stacked LightGBM retrained the same way). The ensemble is
`../ensemble_ctrl_v6`. RESULTS: see the tables at the end and `../ensemble_ctrl_v6/README.md`.

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
