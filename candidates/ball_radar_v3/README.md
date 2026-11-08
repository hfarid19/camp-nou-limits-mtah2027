# ball_radar_v3: ball_radar retrained with the attack direction fixed

Code identical to `ball_radar` except `prep_data.py` and `feat_data.py`, which replace
`learned_dfl_evidence.make_dataset.attack_sign` (home team's mean x over the whole first half) by
the bench's rule (home team over the first 50 frames of the first half, as
`tune_on_sample_data.attack_sign`) before calling `degrade_dfl`, and `feat_data.py` reads the
radar weights from `$MTAH_WORK/arch/ball_radar_v3`. The old rule reversed DFL-MAT-J03WR9 (mean
54.9 m, kick-off 45.7 m), so the radar sequences, the stacked LightGBM features and their canonical
"home attacks +x" frame were mirrored for that game (checked: J03WR9's canonical ball x flips
sign, every other game's training sequences are identical). See `../temporal_tcn_v3/README.md`.

Recipe: ball_radar's final one (README "Rebuild": seeds 31-33, radar 800 iterations, 7 folds +
full, `feat_data.py` for 0-6 + full, `train_lgb.py` d,ha then c), run with 2 torch threads
instead of 1, so the radar weights are not bit-comparable; the script is
`$MTAH_WORK/arch/ball_radar_v3/run_all.sh` (~45 min). The stacked models' LOGO logloss:
team (ha) 0.230 (ball_radar's meta: 0.286), hand-over (c) 0.121 (0.125).

## Results (bench --dfl2, live, LOGO; paired vs ball_radar)

| | raw spellF1 | fine4 | macroF1 |
|---|---|---|---|
| ball_radar | 0.6311 | 0.6643 | 0.9146 |
| **ball_radar_v3** | **0.6445** | **0.6781** | **0.9180** |

fine4 +0.0139 ± 0.0037 (17/21); per game +0.007 +0.016 +0.009 +0.025 +0.027 +0.015, and -0.0005
on J03WR9 (retraining noise there, since the thread count changed).

## Files

`radar.npz` (sha256 `6b161591d2b1`, 0.46 MB), `model_brd.txt` (`9030306222e0`), `model_brha.txt`
(`e0e3a8b5b683`), `model_brc.txt` (`003229528d22`), `model_br_meta.json`; fold models in
`$MTAH_WORK/arch/ball_radar_v3/folds`, mapping in `$MTAH_WORK/arch/ball_radar_v3/folds.json`;
bench outputs in `$MTAH_WORK/probs/ball_radar_v3`.
