# pitch_control

LDE's frame-evidence pipeline (`learned_dfl_evidence`) with **pitch-control features** added to
all three LightGBM models (dead/alive `d`, home/away `ha`, hand-over `c`), retrained with
leave-one-game-out (LOGO) folds on the same domain-randomised DFL degradations. Decoding uses
LDE's round-3 Viterbi decoder and its `CFG`, unchanged.

**Bottom line.** Pitch control adds a small, consistent gain that sits inside the noise band:
DFL LOGO spellF1 **0.6288** against 0.6243 for LDE (+0.0045), Metrica dev 0.6784 against 0.6725.
Held-out log-loss of the team model falls by 0.011. Pitch control does **not** fix the
under-split: the predicted possession counts are almost the same as LDE's (2217 against 3061
true on DFL). On its own, pitch control is a much weaker possession signal than LDE's
nearest-player distance and last-control features. A pitch-control-only H/A model scores 0.5393
on DFL and 0.5712 on dev.

## Method

`pc.py` computes 63 identity-free, swap-aware per-frame features (about 2.5 s per 145k-frame
game, numpy only):

- **Spearman (2018) potential pitch control.** For each player, the time to intercept is
  `tti = t_react + |p + v t_react - x| / v_max`, and the intercept probability is logistic in
  `T - tti` (sigma 0.45 s). Control follows the ODE `dPPCF_j/dT = (1 - sum PPCF) P_int_j(T) lambda`,
  integrated from the ball's arrival time at x. The constants are those of the paper and
  L. Shaw's public implementation: t_react 0.7 s, v_max 5 m/s, lambda 4.3/s. Both teams use the
  same lambda (no attack/defence roles, because those are the unknown).
  - The ODE is solved per frame for the 4 fastest players per team with an exponential
    integrator: `remaining *= exp(-R dt)`, with the drop split in proportion to the rates. This
    is stable at dt = 0.08 s and needs 36 steps.
  - Targets are the ball (short-filled at 0.5 s and interpolated up to 3 s) and the ball's
    actual future positions +0.4, +1 and +2 s ahead. Those use the players at t and that flight
    time ("who wins the ball where it is going").
  - The features also include each team's first and second time to intercept, the tti gap, the
    last and next confident control (> 0.85) with since/until counts, and the control integrated
    over the past and next 1.5 s.
- **Fernandez & Bornn (2018) influence model.** Each player's influence is a bivariate normal
  centred at `p + 0.5 v`, stretched along the velocity by `(v/13)^2`, with a radius of 4 to 10 m
  that grows with distance to the ball. Control is `sigmoid(sum_H I - sum_A I)`. It is evaluated
  at the ball, on a 5 m ring around the ball (mean, and forward-minus-back), and on an 11 x 7
  grid every 5th frame, interpolated. The grid gives home's area share, the share within 20 m of
  the ball, and each team's control of the half it attacks.
- **Multi-scale context** (as in LDE): 0.5 / 2 / 5 s means and a past-minus-future difference
  of the main signals, plus +-0.8 / 2 / 4 s shifts of the Spearman ball control and the tti
  difference.

Velocities come from slot differences, with jumps over 12 m/s dropped and a 5-frame mean. Slots
are compacted to 14 per team. A missing velocity is treated as standing.

## Training data and the integrity checks

- **Data.** Same data as LDE: the 7 open DFL/IDSSE Bundesliga games (Bassek et al., Sci Data
  2025, figshare 28196177 v1, **CC BY 4.0**, credit DFL Deutsche Fussball Liga / IDSSE). Labels
  are the tracking feed's BallStatus / BallPossession flags. Nothing is trained on Metrica or on
  the match.
- **Degradations.** `make_dataset.py` regenerates LDE's degradations exactly, using
  `lde_make_dataset.py` (an unchanged copy) with seeds 11, 12 and 13:
  - r1 (`--drop-p 0 --blackout-p 0`) for `ha`;
  - r2 (the defaults) for `d` and `c`.

  For all 42 game x seed files, the regenerated LDE features were checked **bit-identical** to
  LDE's stored datasets. The bench's own DFL caches (seed 2027) are never used.
- **Training.** `train.py` uses LDE's LightGBM parameters and the same tasks (swap augmentation
  for ha and c). It trains 7 LOGO fold models per model, where the fold index is the position in
  sorted(npz). Rounds are the argmin of the mean held-out log-loss, and the full model is refit
  on all 7 games.
- **Fold mapping.** The folds JSON maps each shipped model's sha256 to its fold models:
  `$MTAH_WORK/arch/pitch_control/folds.json`.
  It also contains LDE's verified entries, which were used for the control runs.

LOGO held-out log-loss (mean of the 7 folds):

| model | LDE (194 features) | LDE + PC (257) | PC only (63) |
|---|---|---|---|
| ha | 0.2754 | **0.2648** | 0.3240 |
| c | 0.1230 | 0.1224 | - |
| d | 0.0586 | 0.0573 | - |

Single-feature sign accuracy on alive DFL frames (J03WMX, seed 11, r1) puts the PC signals
below LDE's distance features:

| feature | sign accuracy |
|---|---|
| `lastctrl` | 0.842 |
| `diff_d1_m2.0` | 0.816 |
| `diff_d1` | 0.769 |
| `pcf_ball_m5.0` | 0.758 |
| `pc_last` | 0.698 |
| `pcs_fut1.0` | 0.687 |
| `tti_d1` | 0.608 |
| `pcs_ball` | 0.599 |

Spearman control at the ball's *current* spot is weak because the 0.7 s velocity projection
carries a running ball carrier past the ball. Control of where the ball is going is the more
useful PC signal.

## Variants (all benched; DFL = `bench --dfl --folds`, seed 2027, fold models)

| variant | DFL spellF1 / macroF1 | dev spellF1 / macroF1 | decision |
|---|---|---|---|
| control: LDE's three models through this pipeline wrapper | 0.6243 / 0.9144 (reproduces LDE exactly) | 0.6725 / 0.8145 | reference |
| PC-only `ha` (63 PC features) + LDE `d`, `c` | 0.5393 / 0.8930 | 0.5712 / 0.7967 | standalone value: well below LDE |
| LDE+PC `ha` + LDE `d`, `c` | 0.6263 / 0.9164 | not run | step kept |
| LDE+PC `ha`, `c` + LDE `d` | 0.6281 / 0.9165 | 0.6779 / 0.8145 | step kept |
| ... + `switch_cost` 0.5 | 0.6269 | | rejected |
| ... + `contrast_beta` 1.5 | 0.6270 | | rejected |
| **LDE+PC `d`, `ha`, `c` (final)** | **0.6288 / 0.9176** | **0.6784 / 0.8144** | shipped |

Final per-game DFL results (spellF1, possessions predicted / true):

| game | spellF1 | possessions pred / true |
|---|---|---|
| J03WMX | 0.5744 | 324 / 511 |
| J03WN1 | 0.5639 | 332 / 500 |
| J03WOH | 0.6783 | 295 / 402 |
| J03WOY | 0.6212 | 327 / 456 |
| J03WPY | 0.6486 | 303 / 421 |
| J03WQQ | 0.6361 | 346 / 401 |
| J03WR9 | 0.6794 | 290 / 370 |

**Holdout** (Metrica Game 3, run once at the end): 0.6804 / 0.8476 (LDE: 0.6845).

**Match.** 342 possessions, mean 9.8 s, dead share 0.470, no X inside the halves. Two
`--match` runs (the second with PYTHONHASHSEED=12345) are byte-identical: md5
9a34b56901375034b9951d33900a4b4e. `validate_submission.py` reports Valid.

All gains over LDE (+0.0045 DFL, +0.006 dev, -0.004 holdout) are below the ~0.04 spellF1 noise
band. Treat this as "LDE with a slightly better team model", not as a new level.

## Ensemble outputs

`predict_proba` returns the evidence the decoder works from:
- P(D) is the dead model plus LDE's long-gap and goal-tag terms;
- P(H | alive) is the swap-symmetric team model after LDE's local-contrast and tag terms;
- P(H) = (1 - P(D)) P(H | alive), and P(A) = (1 - P(D)) (1 - P(H | alive)).

Saved files:
- `$MTAH_WORK/probs/pitch_control/`:
  dev, DFL, holdout and match for the final model.
- `$MTAH_WORK/arch/pitch_control/probs_pconly/`:
  DFL and dev only, for the PC-only variant. This is the more *different* view: its team
  evidence uses no LDE features.

## Files

- `pc.py`: the pitch-control features.
- `features.py`, `lde_pipeline.py`, `lde_make_dataset.py`: unchanged copies of LDE's files.
- `make_dataset.py`, `train.py`, `assemble.py`: build, train and package.
- `pipeline.py`: predict / predict_proba.
- `model_{d,ha,c}.txt` and `model_meta.json`: the shipped models. The meta holds file names,
  feature lists and sha256, and the hashes are asserted at load.

`PC_MODEL_DIR` points the pipeline at another model set. It is for development only and prints
a warning.

## Rebuild

Run from `candidates/pitch_control`, with `S` set to the scratch directory. The DFL npz files
come from LDE's `fetch_idsse.py`.

    PY=$S/venv/bin/python; A=$S/arch/pitch_control
    $PY -I make_dataset.py --npz $S/idsse/npz --variant r1 --out $A/ds_r1 --lde-ds $S/idsse_r1/ds
    $PY -I make_dataset.py --npz $S/idsse/npz --variant r2 --out $A/ds_r2 --lde-ds $S/idsse_r2d/ds
    $PY -I train.py --ds $A/ds_r1 --which ha --feats lde+pc --out $A/m_full --max-rounds 700
    $PY -I train.py --ds $A/ds_r2 --which c  --feats lde+pc --out $A/m_full --max-rounds 600
    $PY -I train.py --ds $A/ds_r2 --which d  --feats lde+pc --out $A/m_full --max-rounds 800
    $PY -I train.py --ds $A/ds_r1 --which ha --feats pc --out $A/m_pconly --max-rounds 700   # PC-only variant
    $PY -I assemble.py --d $A/m_full/model_d.txt --ha $A/m_full/model_ha.txt --c $A/m_full/model_c.txt --out .

Each make_dataset run took about 2 min. Each training ran with one thread, two at a time: ha
about 17 min, c about 14 min, d about 19 min, PC-only ha about 5 min.

Then, from the repo root:

    $PY candidates/bench.py candidates/pitch_control/pipeline.py --dfl --folds $A/folds.json --save-probs $S/probs/pitch_control
    $PY candidates/bench.py candidates/pitch_control/pipeline.py --match --save-probs $S/probs/pitch_control

Requirements: `requirements.txt` (numpy 2.5.4, pandas 3.1.1, lightgbm 4.7.0), the same as LDE.
Inference uses `num_threads=1`.

## References

- W. Spearman (2018). Beyond Expected Goals. MIT Sloan Sports Analytics Conference.
- J. Fernandez, L. Bornn (2018). Wide Open Spaces: A statistical technique for measuring space
  creation in professional soccer. MIT Sloan Sports Analytics Conference.
- L. Shaw, LaurieOnTracking (public Spearman pitch-control implementation), for the parameter
  values.
- M. Bassek, R. Rein, H. Weber, D. Memmert (2025). An integrated dataset of spatiotemporal and
  event data in elite soccer. Scientific Data 12 (figshare 28196177, CC BY 4.0).
