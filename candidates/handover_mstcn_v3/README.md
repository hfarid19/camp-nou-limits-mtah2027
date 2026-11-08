# handover_mstcn_v3: handover_mstcn_v2 (raw target) retrained on sign-fixed data

Same code as `handover_mstcn_v2` (`features.py`, `tcn_np.py` byte-identical; `pipeline.py`
differs only in its docstring), same `prep_data.py --target raw` and `train_tcn.py --cfg
'{"epochs": 15}'` (seed 0), run on `$MTAH_WORK/ds_v2s` instead of `ds_v2`. ds_v2 had
DFL-MAT-J03WR9's attack direction reversed in 3 of its 21 training degradations; see
`../temporal_tcn_v3/README.md` for the bug and `candidates/fix_attack_sign_ds.py` for the fix.

## Results (bench --dfl2, live, LOGO; paired vs handover_mstcn_v2 raw)

| | raw spellF1 | fine4 | macroF1 |
|---|---|---|---|
| handover_mstcn_v2 (raw) | 0.6703 | 0.6961 | 0.9265 |
| **handover_mstcn_v3** | **0.6823** | **0.7080** | **0.9295** |

fine4 +0.0119 ± 0.0029 (16/21); per game +0.017 +0.020 +0.018 +0.005 +0.015 +0.009 and exactly
0.000 on J03WR9 (its fold model sees identical training data and is reproduced bit-for-bit, so it
is a control for retraining noise). Held-out NLL of the fold models on the training seeds drops on
6 of 7 folds (e.g. J03WOH 0.177 -> 0.153).

## Files

| file | contents |
|---|---|
| `model_tcn.npz` | pack of the 7 fold models (outputs averaged), sha256 `459018cb80b4` (9.0 MB) |
| `$MTAH_WORK/arch/handover_mstcn_v3/raw/fold_tcn_<gi>.npz` | LOGO fold models, `train_log.txt` |
| `$MTAH_WORK/arch/handover_mstcn_v3/data_raw`, `folds.json` | prepared sequences; sha -> folds |
| `$MTAH_WORK/probs/handover_mstcn_v3` | bench codes / probabilities |

Rebuild: v2's commands with `--ds $MTAH_WORK/ds_v2s/ds --target raw`, out dir
`$MTAH_WORK/arch/handover_mstcn_v3`, then `pack_model.py <A>/raw <A>/folds.json
candidates/handover_mstcn_v3/model_tcn.npz`.
