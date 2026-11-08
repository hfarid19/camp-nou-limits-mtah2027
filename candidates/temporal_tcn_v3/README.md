# temporal_tcn_v3: temporal_tcn_v2 (raw target) retrained on sign-fixed data

Same code as `temporal_tcn_v2` (`features.py`, `tcn_np.py`, `lde_decoder.py` byte-identical;
`pipeline.py` differs only in its docstring), same training command (`../temporal_tcn_v2/train_tcn.py`,
`--target raw --seed 0`, README of v2), same raw DFL-flag target. Only the data changed.

**The bug.** `make_dataset_v2.py` takes each game's attack direction (`sign`: does the home team
attack +x in the first half?) from `learned_dfl_evidence/make_dataset.attack_sign`, which compares
the home team's mean x over the whole first half with the halfway line. For `DFL-MAT-J03WR9` that
mean is 54.9 m (> 52.5) -> -1, but at the first-half kick-off the home team stands at 45.7 m, i.e. it
attacks +x. The bench (`tune_on_sample_data.attack_sign`, first 50 frames) uses +1, and so does the
match (from its metadata). So 3 of ds_v2's 21 training degradations had every direction-dependent
feature mirrored relative to their labels. The other 6 games agree under both rules.

**The fix.** `candidates/fix_attack_sign_ds.py` builds `$MTAH_WORK/ds_v2s`: every file re-signed
with the bench rule; the 18 unchanged files are hard links, J03WR9's raw npz gets sign +1 and its
features are recomputed (the recompute procedure reproduces a stored ds_v2 file bit-for-bit).

## Results (bench --dfl2, live, LOGO, 7 games x 3 seeds; paired vs temporal_tcn_v2 raw)

| | raw spellF1 | fine4 | macroF1 |
|---|---|---|---|
| temporal_tcn_v2 (raw) | 0.6641 | 0.6931 | 0.9239 |
| **temporal_tcn_v3** | **0.6737** | **0.7028** | **0.9261** |

fine4 +0.0096 ± 0.0040 (15/21 game-seeds); per game +0.011 +0.020 +0.025 +0.010 +0.006 -0.005
and exactly 0.000 on J03WR9: its fold model never trains on J03WR9, so its training data are
unchanged and torch reproduces it bit-for-bit. That fold is a built-in control: the other six
differences come from the fix alone, not from retraining noise.

## Files

| file | contents |
|---|---|
| `tcn_weights.npz` | full model (all 7 games, ds_v2s), sha256 `90b2fe8df174` |
| `$MTAH_WORK/arch/temporal_tcn_v3/raw/fold_tcn_<gi>.npz` | LOGO fold models, `train.log` |
| `$MTAH_WORK/arch/temporal_tcn_v3/folds.json` | `90b2fe8df174` -> raw folds |
| `$MTAH_WORK/probs/temporal_tcn_v3` | bench codes / probabilities |

Rebuild: `python -I candidates/fix_attack_sign_ds.py`, then the v2 command with
`--data $MTAH_WORK/ds_v2s/ds --target raw --seed 0 --fold-dir $MTAH_WORK/arch/temporal_tcn_v3/raw`
and `cp .../raw/full.npz candidates/temporal_tcn_v3/tcn_weights.npz`.
