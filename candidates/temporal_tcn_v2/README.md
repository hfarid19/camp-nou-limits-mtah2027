# temporal_tcn_v2: temporal_tcn retrained on the realistic degradation

This is `temporal_tcn` (a dilated non-causal TCN over LDE's 194 identity-free features, decoded
by LDE's Viterbi decoder; see `../temporal_tcn/README.md`). The architecture, training
arguments (`../temporal_tcn/train_args_v1.json`: C 64, 12 blocks, 800 iterations x 16 crops of
512, lr 2e-3, seed 0), numpy inference, decoder and decoder constants are all unchanged. What is
new is the training data and the training target:

- **Data.** v1 was trained on LDE's round-1 and round-2 degradations (`idsse_r1`, `idsse_r2d`),
  where the ball vanished in many short random bursts during live play. v2 is trained on
  `$MTAH_WORK/ds_v2` (`candidates/make_dataset_v2.py`: the 7 DFL games degraded with a
  randomised `degrade_v2`, seeds 41-43). There the ball is visible for almost all live play and
  missing mostly at dead ball, as in the competition tracking. The bench seeds (2027-2029) are
  never used for training.
- **Target** (`train_tcn.py --target`):
  - `raw`: the DFL flags, as v1.
  - `ctrl`: `y = control.fold(codes, 2.0 s, valid_runs(codes))` on the full-resolution codes of
    `ds_v2/raw/<file>`, subsampled at the same `idx`. The hand-over head target `yc` is
    recomputed from the folded codes. This is the bench's `controlled_labels(gt, 2.0)` applied to
    the training labels. H<->A changes per 1000 steps drop from 11.5 to 6.5.

The code changes from `temporal_tcn` are:

- `--target/--ctrl-s` in `train_tcn.py`;
- the bench wrappers `pipeline_raw.py`, `pipeline_ctrl.py` and `pipeline_ctrl_mix.py`, which
  execute `pipeline.py` under a private name with `WEIGHTS` pointed at the variant's file;
- `paired_codes.py`, the paired comparison tool.

`pipeline.py` differs only in its docstring. `tcn_np.py`, `features.py` and `lde_decoder.py`
are byte-identical to v1's. Inference is numpy only. Weights load through `dfl_folds.resolve`.

**Shipped: `tcn_weights.npz` = `tcn_weights_ctrl.npz`** (ds_v2 only, controlled targets),
sha256 `de6779465699`.

## Variants (bench --dfl2: 7 DFL games leave-one-game-out x 3 degrade_v2 seeds, live runs)

| model | training data | target | ctrl_mean | ctrl T=1 / 2 / 3 | raw spellF1 (SE) | macroF1 | possessions |
|---|---|---|---|---|---|---|---|
| temporal_tcn (v1, shipped member) | r1 + r2d | raw | 0.6708 | 0.6965 / 0.6887 / 0.6272 | 0.6450 (0.017) | 0.915 | 7157 |
| v2 raw | ds_v2 | raw | 0.6790 | 0.7141 / 0.6974 / 0.6255 | **0.6641** (0.014) | 0.924 | 7478 |
| **v2 ctrl (shipped)** | ds_v2 | ctrl | **0.7171** | 0.7078 / 0.7416 / 0.7019 | 0.6429 (0.016) | 0.922 | 6359 |
| v2 ctrl_mix | ds_v2 + r1 + r2d | ctrl | **0.7171** | 0.6993 / 0.7422 / 0.7097 | 0.6344 | 0.919 | 6170 |

Possessions are summed over the 21 game-seeds. The raw truth is 9183.

`ctrl_mix` ties `ctrl` on ctrl_mean (paired -0.0001, SE 0.0025). It loses on raw spellF1
(-0.0086, SE 0.0039, 6/21 wins), and it does no better inside the ensemble (offline vote 0.7458
vs 0.7443, inside the noise). The ds_v2-only model was kept because it is simpler and closer to
the target domain.

### Paired against v1 (same game and seed, 21 pairs; SE over the 7 games with seeds averaged)

| comparison | as predicted | after control.fold(2.0 s) offline on both |
|---|---|---|
| v2 raw - v1, ctrl_mean | +0.0082 (SE 0.0070), 14/21 wins | **+0.0133** (SE 0.0059), 18/21 |
| v2 raw - v1, raw spellF1 | **+0.0190** (SE 0.0054), 17/21 | +0.0112 (SE 0.0061), 16/21 |
| **v2 ctrl - v1, ctrl_mean** | **+0.0463** (SE 0.0042), 21/21 | **+0.0153** (SE 0.0059), 17/21 |
| v2 ctrl - v1, raw spellF1 | -0.0021 (SE 0.0045), 9/21 | +0.0021 (SE 0.0048), 12/21 |
| v2 ctrl_mix - v1, ctrl_mean | +0.0462 (SE 0.0050), 21/21 | +0.0151 (SE 0.0033), 18/21 |

Per-game ctrl_mean gains of the shipped model over v1: +0.061 +0.054 +0.046 +0.044 +0.033 +0.055
+0.032.

What this shows:

- **The realistic training data helps on its own.** With the same raw target, v2 gains +0.019
  raw spellF1, and +0.013 ctrl_mean once both outputs are folded.
- **Relabelled training moves the model to the competition's definition.** Unfolded, the ctrl
  model beats v1 by +0.046 ctrl_mean on every game-seed, and it does not lose on raw spellF1.
- Once an output fold is applied (as `ensemble_ctrl` does), the raw and ctrl variants are
  nearly equal: 0.7335 against 0.7355. The fold on the output recovers most of what the
  relabelled targets give. The data change is the part that survives the fold.

Determinism: the bench of `pipeline.py` reproduces the `pipeline_ctrl.py` run exactly (42 of 42
saved code / probability files byte-identical). Inference is deterministic. Torch training is
seeded, but it is not guaranteed bit-identical across machines or thread counts. The npz files
are the artefacts.

Selection was made on DFL leave-one-game-out (bench v2, ctrl_mean) only. No Metrica, holdout or
competition labels were used, and `evaluation/gt` was never read.

## Files and work locations

| file | contents |
|---|---|
| `tcn_weights.npz` | shipped full model (= `tcn_weights_ctrl.npz`) |
| `tcn_weights_{raw,ctrl,ctrl_mix}.npz` | the variants' full models (trained on all 7 games) |
| `pipeline_{raw,ctrl,ctrl_mix}.py` | bench wrappers for the variants |
| `paired_codes.py` | paired comparison of two saved codes dirs (raw / ctrl_mean, optional offline fold) |
| `$MTAH_WORK/arch/temporal_tcn_v2/{raw,ctrl,ctrl_mix}/fold_tcn_<gi>.npz` | LOGO fold models (gi = index in the sorted DFL list), plus `full.npz`, `train.log`, `train_args_tcn.json` |
| `$MTAH_WORK/arch/temporal_tcn_v2/folds.json` | `de6779465699` -> ctrl, `9714720fa05e` -> raw, `d2f1cb576131` -> ctrl_mix |
| `$MTAH_WORK/probs/temporal_tcn_v2{,_raw,_ctrl,_ctrl_mix}` | saved bench probabilities and codes |

The `ctrl_mix` folds 4, 5, 6 and the full model were trained in separate processes with the same
arguments to save wall time. Each fold reseeds, so this does not change the procedure.

## Rebuild

```sh
W=$MTAH_WORK; PY=$W/venv/bin/python; A=$W/arch/temporal_tcn_v2
$PY -I candidates/make_dataset_v2.py --out $W/ds_v2                  # if not built yet
for V in raw ctrl; do                                                  # ~4 min per model at 2 threads
  OMP_NUM_THREADS=2 $PY -I candidates/temporal_tcn_v2/train_tcn.py --data $W/ds_v2/ds \
      --raw $W/idsse/npz --folds all --fold-dir $A/$V --out $A/$V/full.npz --C 64 \
      --dils 1,2,4,8,16,32,64,128,1,2,4,8 --drop 0.1 --in-drop 0.1 --lr 0.002 --iters 800 \
      --batch 16 --crop 512 --dead-w 1.0 --c-w 0.5 --seed 0 --log-every 200 --threads 2 --target $V
  cp $A/$V/full.npz candidates/temporal_tcn_v2/tcn_weights_$V.npz
done
# ctrl_mix: same with --data $W/ds_v2/ds $W/idsse_r1/ds $W/idsse_r2d/ds --target ctrl, into $A/ctrl_mix
# folds.json: {"<sha256(full.npz)[:12]>": ["$A/<variant>", "tcn"], ...} for each variant
cp candidates/temporal_tcn_v2/tcn_weights_ctrl.npz candidates/temporal_tcn_v2/tcn_weights.npz
$PY candidates/bench.py candidates/temporal_tcn_v2/pipeline.py --dfl2 --folds $A/folds.json \
    --save-probs $W/probs/temporal_tcn_v2
$PY candidates/score_codes.py $W/probs/temporal_tcn $W/probs/temporal_tcn_v2
$PY candidates/temporal_tcn_v2/paired_codes.py $W/probs/temporal_tcn_v2 $W/probs/temporal_tcn [--fold 2.0]
```
