# ball_radar_pff: ball_radar_v3 retrained on PFF World Cup 2022 + DFL

The code is ball_radar_v3's, with fold specs over 55 training games (the 7 DFL games + the 48 PFF
selection games; `brp_common.py`; the 16 PFF holdout games are never included):

- `prep_data.py`: radar sequences for every training game (degrade_dfl, seeds 31-33, kick-off
  sign rule). The DFL sequences are bit-identical to ball_radar_v3's.
- `train_radar.py --spec pff<k>|dfl<g>|full --device cpu|mps`: the same network, loss and
  OneCycle schedule, with 3200 iterations instead of 800 for ~8x the data. Each output has a
  `.games.json` sidecar listing its training games. 4 PFF-fold models were trained on MPS, and the
  DFL folds + full on CPU during a GPU pause.
- `feat_data.py --radar <spec>`: stacked-LightGBM features (degrade_dfl seed 51, stride 5) on the
  radar-imputed ball, for every training game, for each radar spec.
- `train_lgb.py`: fold spec s trains on the games s allows (500 rounds) and records its log-loss
  curve on the games it excludes. The full model's rounds = argmin of the mean of the 11 curves:
  d 235, ha 498, c 481. The params are unchanged.
- `pipeline.py`: `radar.npz` and `model_br{d,ha,c}.txt` load through `pff_folds.resolve`. Under
  run_pff.py a PFF fold booster is cut to the full model's rounds, as bench.py does for DFL folds.

Work files: `$MTAH_WORK/arch/ball_radar_pff/` (seq, feat, folds, logs, the driver scripts
`radar_worker.sh`, `feat_all.sh`, `lgb_rest.sh`, `checks.sh`).

| (alone) | DFL fine4 | vs ball_radar_v3 | PFF out-of-fold fine3 | Metrica dev |
|---|---|---|---|---|
| ball_radar_v3 | 0.6781 | | 0.6538 | 0.7038 |
| ball_radar_pff | 0.6988 | +0.0207 ± 0.0051 (19/21) | 0.7215 | 0.7345 |
