# handover_mstcn

A learned hand-over / boundary detector from the temporal action segmentation literature. An
MS-TCN with an ASRF boundary-regression branch, trained on the open DFL tracking. It predicts,
for every frame, the class (H / A / D) and the probability of a boundary (an H<->A hand-over or
an alive<->dead change). The match is then segmented ASRF-style: it is cut at the boundary
peaks, and each piece takes its mean class probability. There is no HMM and no LightGBM. The
per-frame evidence comes from a temporal network, not from per-frame boosters.

## Results (all from `candidates/bench.py` runs)

| split | spellF1 | macroF1 | possessions pred / true |
|---|---|---|---|
| DFL LOGO (7 games, seed 2027, fold models) | **0.6523** | 0.9202 | 2485 / 3061 |
| Metrica dev (G1, G2 x 2 seeds) | **0.6877** | 0.8105 | 1209 / 1354 |
| Metrica holdout G3 (run once, at the end) | 0.6764 | 0.8365 | 670 / 728 |
| match (no score) | | | 340 possessions, dead share 0.454, mean 10.2 s |

For reference, learned_dfl_evidence (LDE, the current best) scores 0.6243 / 0.9144 on DFL LOGO
with 2202 / 3061 possessions, 0.6725 on dev and 0.6845 on the holdout.

DFL per game (spellF1, poss pred/gt): WMX 0.6037 347/511, WN1 0.5975 365/500, WOH 0.6988
333/402, WOY 0.6502 377/456, WPY 0.6741 373/421, WQQ 0.6660 366/401, WR9 0.6755 324/370.

- **DFL LOGO:** +0.028 over LDE. That is above the roughly 0.01 noise band for 3061 spells,
  and macroF1 is also up (+0.006).
- **Metrica dev:** +0.015 over LDE. Dev was not used for any choice.
- **Holdout:** -0.008 against LDE, inside the noise band.
- **Under-splitting:** it is reduced, not fixed. DFL counts go from 2202 to 2485 of 3061, and
  the match from 325 to 340. Short spells are still the main miss.

## Method

References:
- Y. A. Farha, J. Gall. *MS-TCN: Multi-Stage Temporal Convolutional Network for Action
  Segmentation.* CVPR 2019. Dilated residual stages, refinement stages on the previous stage's
  softmax, CE + truncated-MSE smoothing loss (tau 4, lambda 0.15).
- Y. Ishikawa, S. Kasai, Y. Aoki, H. Kataoka. *Alleviating Over-segmentation Errors by Detecting
  Action Boundaries* (ASRF). WACV 2021. A shared backbone with an action segmentation branch and
  a boundary regression branch. Boundaries are the local maxima of the boundary probability
  above a threshold, and each resulting segment is labelled by its frame-wise class predictions.

**Input.** The 194 identity-free, team-relative per-frame features of learned_dfl_evidence
(`features.py`, copied unchanged). They are standardised with leave-one-game-out statistics,
computed symmetrically over team-swapped copies, and NaN becomes 0.

**Time base.** The network runs at 5 fps, on one stride-5 phase of the 25 fps frames. This
is exactly the layout of LDE's `make_dataset.py` rows. At inference all 5 phases of each half
are run and interleaved back to 25 fps, so the boundary probability has frame resolution.

**Network** (`train_tcn.py`, about 0.3 M parameters):
- Backbone: 1x1 conv 194 -> 64, then 10 dilated residual layers (dilation 2^i, kernel 3,
  acausal, dropout 0.5), so the receptive field is about 410 s. Two heads: 3 class logits and
  2 boundary logits (H<->A, alive<->dead).
- Class branch: 2 MS-TCN refinement stages (32 channels, 10 layers) on the softmax.
- Boundary branch: 1 ASRF refinement stage on the sigmoid.

**Loss.** Every stage gets CE + 0.15 * T-MSE, and every boundary stage gets BCE with
pos_weight 10. The boundary targets are Gaussian (sigma 2.5 frames) around each tracking-flag
change, placed half-way between the two frames, so a 5 fps sample still knows the change's
sub-sample position.

**Training.** Adam 5e-4, crops of 1024 samples (205 s), batch 8, 600 crops per epoch, 15
epochs, team-swap augmentation on half of the crops (features via `swap_index`, H<->A labels
swapped). Inference averages the prediction with the team-swapped prediction, so H/A are
exactly symmetric.

**Folds.** There are 7 leave-one-game-out models (`fold_tcn_<gi>.npz`, gi = position in the
sorted DFL match list), about 3-4 min each with 2 threads. The shipped `model_tcn.npz` packs
all 7 and averages their outputs (`pack_model.py`), so no extra full-data model is trained.
- Under `bench --dfl`, `dfl_folds.resolve()` replaces the pack with the single fold model that
  never saw the scored game.
- The DFL score is therefore for single fold models. The Metrica and match runs use the
  7-model average.

**Inference.** Plain numpy (`tcn_np.py`), with no torch and no LightGBM at inference. The
numpy forward was checked against torch: max abs difference 6e-7.

**Decoder** (`pipeline.decode`, CFG):
1. **Local contrast.** LDE's local-contrast term on the team logit
   log(P(H)/P(A)), with beta 1.25, gate 1.25 and a 100-frame half width. It lets short
   opposing dips cross 0.
2. **ASRF segmentation.** Find the peaks of max(b_ha, b_dead) at or above 0.2 that are the
   maximum within +-0.4 s. Cut there, and label each piece with argmax of its mean class
   probability.
3. **Boundary-forced hand-over.** Take an alive piece of at most 2 s that is cut out by two
   boundary peaks of at least 0.5 but labelled like both neighbours. It flips to the other
   team if that team holds at least 0.3 of its alive mass. This is the explicit
   under-splitting fix.

`predict_proba` returns the final-stage class posteriors (P(H), P(A), P(D)), team-swap
symmetrised, before the decoder.

## Variants (DFL LOGO spellF1, fold models; harness `tuning/logo_cache.py`, which reproduces bench --dfl to 1e-4)

| variant | DFL spellF1 | poss (of 3061) | decision |
|---|---|---|---|
| v1 model + 3-state Viterbi on the class log-probs, static costs | 0.6351 | 2199 | baseline |
| + boundary probs as dynamic switch costs, c(t) = c0 - gamma * (logit b(t) - logit 0.05). gamma 1, c0 3 | 0.5655 | 3100 | over-splits |
| same, gamma 0.5, c0 7 (best of the c0 5/7/9 x gamma 0.25/0.5 grid) | 0.6391 | 2211 | |
| ASRF segmentation, threshold 0.5, window 0.4 s | 0.6352 | 2122 | |
| ASRF, threshold 0.2, window 0.2 s (0.15-0.25 x 0.2-0.4 s plateau 0.643-0.6455; 0.8 s window worse) | 0.6453 | 2263 | kept |
| separate H<->A / dead thresholds (0.2-0.3 x 0.2-0.5) | 0.6426-0.6449 | | rejected |
| segment label by mean log-prob | 0.6446 | | rejected |
| boundary-prob smoothing (half width 2 / 4 frames) | 0.6452 / 0.6452 | | rejected |
| LDE/v2 dead-edge post chain (gap-end snap 1 s, out look-back 6 s) | 0.6446 | | rejected (the TCN's D edges are already learned) |
| + local-contrast team logit, beta 1.25, gate 1.25, hw 100 (window 0.2 s / 0.4 s) | 0.6503 / 0.6514 | 2386 / 2335 | kept (window 0.4 s) |
| contrast beta 1.5 / 2.0, gate 1.0 / 0.75, hw 50 / 200 | 0.6449-0.6525 | | plateau, kept 1.25 / 1.25 / 100 |
| + boundary-forced hand-over, peak 0.5, share 0.3, at most 2 s | **0.6522** (bench 0.6523) | **2485** | kept: equal score, +150 possessions toward truth |
| same, peak 0.3 / share 0.2 / 0.3 / 0.4; peak 0.5 / share 0.2 / 0.4; at most 4 s | 0.6309-0.6505 | 2393-2785 | rejected (share 0.2 fragments) |
| v2 model: T-MSE lambda 0.05, 10 epochs (otherwise v1); best decoder | 0.6320 | 2134 | rejected (-0.02) |

The boundary-forced flip is +0.0008, which is noise. It was kept because it moves the count
toward the truth at no cost to the score. That count is the explicit goal of this candidate,
and turning it off would cost nothing measurable either. Training used about 45 min of wall
time with 2 threads (v1 28 min, v2 17 min, on a shared machine), so no further retrains (more
data, other losses) were tried.

What did not work, plainly:
- The boundary branch on its own detects about 60% of the true H<->A changes within +-5 frames,
  at a peak precision of about 0.2 (threshold 0.2). Dead changes: about 64% recall.
- Feeding the boundary probabilities into a Viterbi as dynamic switch costs either over-splits
  badly (0.566 with 3100 spells) or does no better than static costs.
- What works is the ASRF cut-then-vote refinement. The many false peaks are harmless there,
  because a cut between two pieces with the same label disappears.

## Data and licences

- Training labels: the open DFL / IDSSE dataset (Bassek, Rein, Weber, Memmert 2025, *Sci Data*
  12, figshare 28196177 v1, CC BY 4.0, (c) DFL Deutsche Fussball Liga / IDSSE). These are the
  per-frame BallStatus / BallPossession flags of the tracking feed for 7 Bundesliga games,
  fetched and md5-checked by `learned_dfl_evidence/fetch_idsse.py`.
- No hand labels, no Metrica labels and nothing from the competition match are used in training.
- Training inputs: LDE's domain-randomised degradations (`learned_dfl_evidence/make_dataset.py`,
  round-2 settings, degradation seeds 11 / 12 / 13). These are not the bench's own DFL caches
  (seed 2027), which were only scored, through bench.py and the identical
  `tuning/logo_cache.py`.
- Decoder constants were chosen on DFL LOGO only. Metrica dev was run once with the final
  config, and the holdout was run once at the end. evaluation/gt was never read.

## Rebuild

    pip install -r candidates/handover_mstcn/requirements.txt
    sh candidates/handover_mstcn/rebuild_models.sh <IDSSE_CACHE> <OUT>
    #   = learned_dfl_evidence/make_dataset.py --cache <IDSSE_CACHE>/r2   (round-2 settings)
    #     prep_data.py --ds <IDSSE_CACHE>/r2/ds --npz <IDSSE_CACHE>/npz --out <OUT>/data
    #     OMP_NUM_THREADS=2 train_tcn.py --data <OUT>/data --out <OUT>/folds --folds 0,...,6 --cfg '{"epochs": 15}'
    #     pack_model.py <OUT>/folds <OUT>/folds.json      (writes model_tcn.npz + the folds JSON)
    python candidates/bench.py candidates/handover_mstcn/pipeline.py --dfl --folds <OUT>/folds.json
    python candidates/bench.py candidates/handover_mstcn/pipeline.py --match   # writes submission.csv

Notes on the rebuild:
- Torch CPU training is seeded but has not been verified to be bit-reproducible across
  machines. The committed `model_tcn.npz` (9 MB, float32) is the artefact the submission
  depends on.
- The match submission is byte-identical across two `--match` runs (checked with `cmp`, and so
  is `predict_proba`).
- Inference takes about 5 s per DFL game and about 20 s per Metrica game.
- In this workspace the fold models live in `scratchpad/arch/handover_mstcn/v1/` and the folds
  JSON is `scratchpad/arch/handover_mstcn/folds.json`.
