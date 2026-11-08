# handover_mstcn_v2: handover_mstcn retrained on the realistic degradation

This is `handover_mstcn` (ASRF-style MS-TCN, see `../handover_mstcn/README.md` for the method)
with the **same architecture, loss, training settings, decoder and decoder constants**. The
training data and the training target are new:

- **Data.** The old model was trained on LDE's round-2 degradation (`idsse_r2d`), where the ball
  vanished in many short random bursts during live play. This one is trained only on
  `$MTAH_WORK/ds_v2` (`candidates/make_dataset_v2.py`: the 7 DFL games degraded with a
  randomised `degrade_v2`, seeds 41-43). There the ball is visible for almost all live play and
  missing mostly at dead ball, as in the competition tracking. It is the same amount of data as
  before (21 degradations). The bench seeds (2027-2029) are never used for training.
- **Target** (`prep_data.py --target`). There are two variants:
  - `raw`: the DFL BallStatus / BallPossession flags, as before.
  - `ctrl`: the challenge's controlled-possession rule applied to the labels before any target
    is derived. `control.fold(codes, 2.0 s, valid_runs)` runs on the full-resolution 25 fps
    codes (checked equal to `ds_v2/raw/<file>`'s `codes`), so every H/A spell shorter than 2 s
    that follows the other team's possession is handed back to that team. The class targets
    and both boundary targets (H<->A, alive<->dead) then come from the folded codes at the
    same stride-5 `idx`. This removes 44% of the H<->A changes (2351 -> 1307 over the 7 games;
    J03WMX 408 -> 195).

The code changes from `handover_mstcn` are:

- the `--target/--ctrl-s` option in `prep_data.py`;
- an output path argument in `pack_model.py`;
- `pipeline_raw.py` / `pipeline_ctrl.py`, bench wrappers that point `pipeline.py` at
  `model_tcn_raw.npz` / `model_tcn_ctrl.npz`.

`pipeline.py`, `tcn_np.py`, `train_tcn.py` and `features.py` are byte-for-byte copies apart from
docstrings. Inference is plain numpy. The weights load through `dfl_folds.resolve`.

**Shipped: `model_tcn.npz` = `model_tcn_ctrl.npz`** (the variant with the higher ctrl_mean).

## Results (bench --dfl2: 7 DFL games leave-one-game-out x 3 degrade_v2 seeds, live runs)

| model | ctrl_mean | ctrl T=1 / 2 / 3 | raw spellF1 (SE) | macroF1 | possessions |
|---|---|---|---|---|---|
| handover_mstcn (old, shipped member) | 0.6776 | 0.7005 / 0.6944 / 0.6380 | 0.6515 (0.016) | 0.917 | 7029 |
| v2 `raw` target | 0.6754 | 0.7124 / 0.6876 / 0.6263 | **0.6703** (0.015) | 0.927 | 7558 |
| **v2 `ctrl` target (shipped)** | **0.7269** | 0.6953 / 0.7507 / 0.7347 | 0.6251 (0.015) | 0.921 | 5945 |

Possessions are summed over 21 game-seeds. The truth is 9183 raw, and fewer under the control
rule.

Paired against the old member (same game and seed, 21 pairs; SE over the 7 games with seeds
averaged; `temporal_tcn_v2/paired_codes.py`):

| comparison | as predicted | after control.fold(2.0 s) offline on both |
|---|---|---|
| v2 raw - old, ctrl_mean | -0.0022 (SE 0.0045), 9/21 wins | **+0.0148** (SE 0.0046), 16/21 |
| v2 raw - old, raw spellF1 | **+0.0188** (SE 0.0051), 19/21 | +0.0111 (SE 0.0046), 16/21 |
| v2 ctrl - old, ctrl_mean | **+0.0493** (SE 0.0077), 21/21 | **+0.0152** (SE 0.0052), 17/21 |
| v2 ctrl - old, raw spellF1 | -0.0263 (SE 0.0069), 3/21 | -0.0116 (SE 0.0039), 5/21 |

What this shows:

- Training on the realistic degradation helps by itself. The raw-target model gains +0.019
  raw spellF1 on the every-touch labels it was trained for, and +0.015 ctrl_mean once its output
  is folded.
- Training on folded targets moves the model to the controlled-possession definition directly.
  Unfolded it gains +0.049 ctrl_mean (all 21 pairs). After an output fold, the two v2 variants
  score the same: ctrl_mean 0.7351 (ctrl) against 0.7347 (raw). The control rule on the output,
  as `ensemble_ctrl` applies it, therefore recovers most of what the relabelled training gives.
  The data change is the part that survives the fold.
- Per-game ctrl_mean gains of the shipped model over the old one: +0.041 +0.042 +0.029 +0.036
  +0.090 +0.049 +0.058.

Determinism: the bench of `pipeline.py` (`model_tcn.npz`) reproduces the `pipeline_ctrl.py` run
exactly (42 of 42 saved code / probability files byte-identical). Torch training is seeded but
not guaranteed to be bit-identical across machines. The npz files are the artefacts.

Selection used DFL leave-one-game-out only (ctrl_mean, bench v2). No Metrica, holdout or
competition labels were used. `evaluation/gt` was never read.

## Files and work locations

| file | contents |
|---|---|
| `model_tcn.npz` | shipped pack of the 7 `ctrl` fold models (outputs averaged; sha256 `eb403d9398bc`) |
| `model_tcn_ctrl.npz`, `model_tcn_raw.npz` | the two variant packs |
| `pipeline_ctrl.py`, `pipeline_raw.py` | bench wrappers for the variants |
| `$MTAH_WORK/arch/handover_mstcn_v2/{raw,ctrl}/fold_tcn_<gi>.npz` | LOGO fold models (gi = index in the sorted DFL list), with `train_log.txt` |
| `$MTAH_WORK/arch/handover_mstcn_v2/folds.json` | `eb403d9398bc` -> ctrl folds, `49004eb6b09d` -> raw folds |
| `$MTAH_WORK/arch/handover_mstcn_v2/data_{raw,ctrl}` | prepared training sequences |
| `$MTAH_WORK/probs/handover_mstcn_v2{,_raw,_ctrl}` | saved bench probabilities and codes |

As in `handover_mstcn`, the shipped model is the 7-fold pack (the average of the 7 leave-one-game-out
models, which together have seen every game), not a separately trained full model. Under the
bench, `dfl_folds.resolve` swaps it for the single fold model that never saw the scored game.

## Rebuild

```sh
W=$MTAH_WORK; PY=$W/venv/bin/python; A=$W/arch/handover_mstcn_v2
$PY -I candidates/make_dataset_v2.py --out $W/ds_v2                    # if not built yet
for V in raw ctrl; do
  $PY -I candidates/handover_mstcn_v2/prep_data.py --ds $W/ds_v2/ds --npz $W/idsse/npz \
      --out $A/data_$V --target $V
  OMP_NUM_THREADS=1 $PY -I candidates/handover_mstcn_v2/train_tcn.py --data $A/data_$V \
      --out $A/$V --folds 0,1,2,3,4,5,6 --threads 1 --cfg '{"epochs": 15}'   # ~2.5 min per fold
  $PY -I candidates/handover_mstcn_v2/pack_model.py $A/$V $A/folds_$V.json \
      candidates/handover_mstcn_v2/model_tcn_$V.npz
done
# folds.json = union of folds_raw.json and folds_ctrl.json
cp candidates/handover_mstcn_v2/model_tcn_ctrl.npz candidates/handover_mstcn_v2/model_tcn.npz
$PY candidates/bench.py candidates/handover_mstcn_v2/pipeline.py --dfl2 --folds $A/folds.json \
    --save-probs $W/probs/handover_mstcn_v2
$PY candidates/score_codes.py $W/probs/handover_mstcn $W/probs/handover_mstcn_v2
$PY candidates/temporal_tcn_v2/paired_codes.py $W/probs/handover_mstcn_v2 $W/probs/handover_mstcn [--fold 2.0]
```
