# nn_pff_tcn: the TCN member retrained on PFF World Cup 2022 + DFL (numpy inference)

`pipeline.py`, `features.py`, `lde_decoder.py` and `tcn_np.py` are temporal_tcn_v3's. There are
three changes. The weights load through `pff_folds.resolve`: the DFL leave-one-game-out fold
under bench --dfl2, or the PFF out-of-fold model under `../nn_pff/run_pff.py`. They are reloaded
on every call rather than cached across games. `WEIGHTS` may also be a list of files, and then
the softmaxes are averaged (seed ensembles); with a single file the outputs are bit-identical to
before. The decoder and its constants are unchanged.

| file | contents |
|---|---|
| `pipeline_v5cpff.py` | the v5c+PFF member: weights `tcn_v5cpff.npz` = `$MTAH_WORK/arch/nn_pff/t5c/full.npz` (sha256 `0cb7703aa75d`) |
| `pipeline_cfg.py` | bench wrapper for any trained config (`NN_PFF_MODEL=<config>/full.npz`; experiments only) |
| `pff_folds.py` | copy of `../nn_pff/pff_folds.py` |

Training: `../nn_pff/train_tcn.py` (recipe, data, folds and results in `../nn_pff/README.md`).
