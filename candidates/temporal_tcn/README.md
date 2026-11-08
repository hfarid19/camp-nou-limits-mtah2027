# temporal_tcn — dilated temporal CNN over identity-free frame features

A sequence model replaces LDE's per-frame LightGBM evidence. A non-causal temporal convolutional
network (TCN) reads ~108 s of context and outputs per-frame P(H), P(A), P(D) plus a hand-over
probability. The output then goes through **exactly LDE's decoder** (Viterbi + snapping + post
chain, copied verbatim into `lde_decoder.py`). That makes the comparison with LDE fair: the
evidence is the only thing that differs.

## Method

* **Inputs.** The 194 identity-free, team-relative per-frame features of LDE's `features.py`
  (copied unchanged). They are standardised (mean/sd from the training games and their H/A
  mirrors, clipped to ±6), NaN is set to 0, and a missing-value indicator is added for every
  column that is NaN on more than 0.5% of rows.
* **Time resolution.** The network runs at 5 fps on point-sampled frames (t ≡ p mod 5). At
  inference all 5 phases p = 0..4 are run and interleaved, so every frame at 25 fps gets its own
  prediction and nothing is interpolated.
* **Architecture** (`train_tcn.py`, C = 64, about 220k parameters):
  * a 1×1 input projection;
  * 12 pre-activation residual blocks `x + W_b·relu(conv3_dil_d(relu(x)))`, with dilations
    1,2,4,…,128,1,2,4,8 and zero padding, non-causal. The receptive field is 541 steps
    (≈108 s, ±54 s);
  * 4 output logits: H, A, D, and hand-over (a direct H↔A flag change within ±4 frames).
  * Dropout is 0.1 on the input and inside the blocks.
* **Loss.** 3-class cross-entropy plus 0.5 × BCE on the hand-over head.
* **Training.** AdamW with lr 2e-3 on a one-cycle schedule, 800 iterations × 16 random 512-step
  crops (102 s each), 2 threads, about 6 min per model.
* **H/A symmetry.** Training uses team-swap augmentation: with p = 0.5 a crop is replaced by its
  mirror (`features.swap_index`, labels H↔A). Inference averages the original and the mirrored
  pass, so H/A are exactly symmetric.
* **Inference.** Plain numpy (`tcn_np.py`): no torch and no lightgbm at inference.
  Deterministic: two `--match` runs give byte-identical CSVs and probability files.
* **`predict_proba`** returns the TCN's own mirror-averaged softmax. It is not one-hot (47% of
  match frames have max prob > 0.99).
* **Decoder input.** `p_dead = P(D)` and `p_home = P(H)/(P(H)+P(A))`. The hand-over head drives
  `snap_to_peak`. All of LDE's decoder constants are left at their defaults (switch 0.75, dead 5,
  contrast 1.25, w_gap 3, refine 0.5 s, min_poss 1 s).
* **Raw mode** (`CFG['mode']=1`, comparison only): per-frame argmax, then a 1 s centred majority
  filter, then `merge_short_v2` (min 1 s).

References:
* Bai, Kolter & Koltun (2018), "An Empirical Evaluation of Generic Convolutional and Recurrent
  Networks for Sequence Modeling" (TCN; dilated residual convolutions).
* Lea et al. (2017), "Temporal Convolutional Networks for Action Segmentation and Detection".
* Farha & Gall (2019), "MS-TCN" (non-causal dilated stacks for frame-wise segmentation).

## Data and licences

* **Training data.** The 7 open DFL/IDSSE Bundesliga games: Bassek, Rein, Weber & Memmert (2025),
  *Scientific Data* 12, figshare 28196177, © DFL/IDSSE, **CC BY 4.0**. They are fetched by
  `learned_dfl_evidence/fetch_idsse.py`.
* **Labels.** From the tracking feed's own BallStatus/BallPossession flags.
* **Degradations.** LDE's `make_dataset.py` domain-randomised degradations with seeds 11/12/13:
  * the round-1 set (`--drop-p 0 --blackout-p 0`);
  * the round-2 set (defaults).

  That makes 42 degradations × 2 halves = 84 sequences and 1.2 M steps. The bench's own
  degradation (seed 2027) is never used for training.
* **Not used:** no Metrica data and nothing from the competition match is used for training.
* **Leave-one-game-out.** `fold_tcn_<gi>.npz` is trained without game gi (gi = index in the
  sorted DFL match list). `tcn_weights.npz` is trained on all 7 games. The weights are loaded
  through `dfl_folds.resolve`. The folds JSON is
  `$MTAH_WORK/arch/temporal_tcn/folds.json`
  (sha256 `dd82eb75578b…` → `arch/temporal_tcn/v1`, kind `tcn`). This candidate does not load
  any LDE model.

## Results (all from `candidates/bench.py`)

| run | spellF1 | macroF1 | possessions pred/gt |
|---|---|---|---|
| **DFL LOGO (7 games, seed 2027)** | **0.6544** | **0.9230** | 2467/3061 |
| LDE on DFL LOGO, same decoder (reference) | 0.6243 | 0.9144 | 2202/3061 |
| TCN raw argmax + 1 s mode filter + merge (DFL) | 0.6487 | 0.9248 | 2268/3061 |
| Metrica dev (G1, G2 × 2 seeds) | 0.6837 | 0.8140 | 318,328,296,300 / 348,348,329,329 |
| TCN raw mode, Metrica dev | 0.6655 | 0.8082 | |
| Metrica holdout G3 (run once, at the end) | 0.6793 | 0.8409 | 335,342 / 364 |
| LDE references: dev 0.6725, holdout 0.6845 | | | |

DFL per game, TCN vs LDE spellF1 (same decoder). The TCN is better on all 7 games, and it
predicts 10–20% more possessions:

| game | TCN | LDE | TCN poss | LDE poss | gt |
|---|---|---|---|---|---|
| J03WMX | 0.6018 | 0.5575 | 357 | 317 | 511 |
| J03WN1 | 0.5940 | 0.5776 | 364 | 313 | 500 |
| J03WOH | 0.6886 | 0.6617 | 345 | 295 | 402 |
| J03WOY | 0.6415 | 0.6200 | 382 | 324 | 456 |
| J03WPY | 0.6651 | 0.6408 | 350 | 310 | 421 |
| J03WQQ | 0.6771 | 0.6415 | 367 | 346 | 401 |
| J03WR9 | 0.7125 | 0.6711 | 302 | 297 | 370 |

Frame accuracy (argmax, DFL OOF) is 0.925–0.952 for the TCN against 0.918–0.937 for LDE on
every game. The TCN makes fewer errors in both directions: team (H↔A) errors and dead/alive
errors.

Match (`submission.csv`): 382 possessions, mean 8.8 s, dead share 0.470, no X inside the halves.
Two runs are byte-identical (CSV and `match.npy`).

The TCN still under-splits, but less than LDE: 2467 predicted spells against 3061 true on DFL,
where LDE predicts 2202.

## Variants (DFL LOGO spellF1; "offline" = `oof_select.py`)

`oof_select.py` replicates `bench --dfl` on the same cached degradation, with fold model gi for
game gi. It reproduces the bench exactly for both LDE (0.6243) and v1 (0.6544).

| step | DFL | kept |
|---|---|---|
| v1: TCN as above + LDE decoder at defaults | **0.6544** (bench) | yes, final |
| v1 + raw argmax/mode-filter instead of the decoder | 0.6487 (bench) | no |
| v1 decoder: switch 0.5/1.0 × dead 3/5/7 | 0.6530–0.6551 | no (flat plateau) |
| v1 decoder: contrast_beta 0 / 0.6 | 0.6406 / 0.6510 | no (the contrast still helps) |
| v1 decoder: w_gap 0 / 1.5 | 0.6571 / 0.6561 | no (+0.003, within noise; defaults kept) |
| v1 decoder: min_poss 0.5 / 0.75 s | 0.6545 / 0.6547 | no (more spells, no gain) |
| v1 decoder: refine window 0.25 / 1.0 s; first-hit snap | 0.6540 / 0.6501; 0.6510 | no |
| v1 snapping on LDE's model_c hand-over detector instead of the TCN head | 0.6558 | no (noise; stays self-contained) |
| v2: + transition-weighted CE (×3 within ±0.4 s of any label change) | 0.6473 (offline) | no |
| v1 + v2 probability average | 0.6494 (offline) | no |

Diagnostic only, not part of this candidate: a geometric mean of TCN and LDE probabilities on 4
DFL games scored 0.6381, against 0.6315 for the TCN alone. That suggests some complementarity
for the ensemble.

## Rebuild

```sh
PY=python   # the pinned venv; OMP_NUM_THREADS=2
# 1. data: LDE's fetch + the two LDE degradation sets (see learned_dfl_evidence/rebuild_models.sh)
$PY -I candidates/learned_dfl_evidence/fetch_idsse.py --games 7 --cache $C
$PY -I candidates/learned_dfl_evidence/make_dataset.py --cache $C/r1 --drop-p 0 --blackout-p 0
$PY -I candidates/learned_dfl_evidence/make_dataset.py --cache $C/r2
# 2. 7 LOGO folds + the full model (≈ 50 min at 2 threads on a loaded machine)
OMP_NUM_THREADS=2 $PY -I candidates/temporal_tcn/train_tcn.py --data $C/r1/ds $C/r2/ds \
    --raw $C/npz --folds all --fold-dir $FOLDS --out candidates/temporal_tcn/tcn_weights.npz \
    --iters 800 --log-every 200
# 3. folds.json: {"<sha256(tcn_weights.npz)[:12]>": ["$FOLDS", "tcn"]}
# 4. bench
$PY candidates/bench.py candidates/temporal_tcn/pipeline.py --dfl --folds folds.json
$PY candidates/bench.py candidates/temporal_tcn/pipeline.py --match
```

The exact v1 arguments are in `train_args_v1.json`. Training with torch is not guaranteed to be
bit-reproducible across machines or thread counts. Inference from the shipped `tcn_weights.npz`
is deterministic.

## Files

| file | contents |
|---|---|
| `pipeline.py` | `predict` and `predict_proba` |
| `tcn_np.py` | numpy inference |
| `train_tcn.py` | torch training and export |
| `features.py` | copy of LDE's |
| `lde_decoder.py` | verbatim copy of LDE's pipeline, used only for its decoder |
| `oof_select.py` | offline LOGO decoder sweeps |
| `tcn_weights.npz` | full model, 0.9 MB |
| `submission.csv` | the match submission |
