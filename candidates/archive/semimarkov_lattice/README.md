# semimarkov_lattice — semi-Markov (explicit-duration) decoding over a touch lattice

Every candidate built so far under-splits: about 300-350 possessions predicted per DFL game against
370-511 true ones. One suspected cause is that frame-level Viterbi implicitly gives spells a
geometric duration. This candidate keeps learned_dfl_evidence's (LDE) per-frame evidence and
replaces its frame-level Viterbi decoder with a segmental (hidden semi-Markov) decoder. That
decoder has explicit, learned duration distributions, and its segment boundaries are restricted
to a lattice of likely ball touches.

**Bottom line (DFL LOGO spellF1, the selection metric):** 0.6422 (macroF1 0.9160). LDE scores
0.6243 / 0.9144. The gap is +0.018, which is below the ~0.04 noise band EVALUATION.md quotes for a
full match. Most of that gain comes from an *evidence* change, not from the decoder (see "What
the numbers say"). The explicit duration model is not what helps. At the final evidence, a
geometric duration pmf with the same mean scores the same (0.6405 vs 0.6409). Under-splitting is
reduced (2417 vs ~2200 predicted spells out of 3061 true) but not solved. Short true spells
(< 2 s) are still mostly missed, because the evidence does not see them.

| bench | spellF1 | macroF1 | possessions pred/gt |
|---|---|---|---|
| DFL LOGO (7 games, seed 2027, fold models) | **0.6422** | 0.9160 | 2417 / 3061 |
| Metrica dev (Games 1+2 x 2 seeds) | 0.6751 | 0.8149 | 1244 / 1354 |
| Metrica holdout Game 3 (run once, at the end) | 0.6835 | 0.8480 | 663 / 728 |
| LDE for reference: DFL / dev / holdout | 0.6243 / 0.6725 / 0.6845 | 0.9144 / 0.8118 / – | |

Match (BAR-ATH): 388 possessions, mean 8.6 s, dead share 0.473 (LDE: 325 possessions). Two
`--match` runs are byte-identical (submission.csv and the saved match probabilities, both checked
with `cmp`).

## Method

1. **Evidence (reused, unchanged models).** `lde_base.py` is a verbatim copy of
   `learned_dfl_evidence/pipeline.py`, and `features.py`, `model_{d,ha,c}.txt` and
   `model_meta.json` are copies of its files (sha256 7ad6ba2e28cb ha, f62083d80f84 d,
   c5011bcf416f c). That gives three LightGBM boosters trained on the 7 DFL games with domain
   randomisation: P(dead), P(home | alive) and P(direct hand-over within +-4 frames). They are
   loaded with `lightgbm.Booster(model_file=...)`, so bench --dfl swaps in the verified
   leave-one-game-out fold models (folds_verified.json). The evidence construction is LDE's: the
   long-ball-gap dead term, ball-out frames, and the Smart Tagging priors (match only). The one
   change is the local-contrast team logit, `l + beta * deadzone(l - moving_mean(l, 100), gate)`,
   with beta 1.0 and gate 0.25 instead of LDE's 1.25 / 1.25. Per-frame scores are
   `tau * [log pH + log(1-pD), log(1-pH) + log(1-pD), log pD]` with tau 0.3.
2. **Touch lattice** (`lattice`, `touch_points`). Candidate boundary frames are: local peaks of
   the hand-over detector (p > 0.05), sign flips and jumps (|Δ| > 1 over 4 frames) of the smoothed
   dead logit, raw-ball gap edges (gaps >= 5 frames), ball-out flips, ball kinematic events (a
   velocity change > 3 m/s between the 3-frame windows before and after a frame, ball moving
   > 1.5 m/s), nearest-player distance minima under 2 m, and changes of which team is nearest.
   A fallback grid every 0.5 s is added, and points closer than 2 frames are merged. That makes
   roughly 3-4 points per second.
3. **Segmental DP** (`build_model`, `semimarkov`). States are H, A, D|H and D|A, where D
   remembers which team had the ball before the stoppage. A segment between lattice points i < j
   in state s scores:
   - its summed tempered frame evidence (cumulative sums, O(1) per segment);
   - plus `w_dur * log p_class(duration)`, the learned duration log-pmf;
   - plus a per-possession bonus (1.0).

   Transitions score `w_trans * log P`. The learned P(hand-over | possession ends) is 0.77 and
   P(restart by the other team | stoppage) is 0.70. On H<->A boundaries a further
   `gamma * (logit p_change - logit prior)` term (max over +-2 frames) is added, with gamma 0.6.
   Viterbi is exact over all segments up to 150 s (possession) / 400 s (stoppage). A same-state
   continuation costs 12 and exists only so longer spells stay representable. A 10-state variant
   (`ctx=1`) gives each possession a duration pmf that depends on its context (entered from a
   restart or a hand-over; left to a stoppage or a hand-over). It was tried and not kept (see
   below).
4. **Durations and transitions** (`train_durations.py`). Spells are cut from the DFL tracking
   flags (D where BallStatus 0, else BallPossession; the bench's rule). Durations use a Gaussian
   KDE in log-seconds (bandwidth 0.25 at 1000 spells, scaled by n^-1/5), normalised over integer
   frame counts 1..10000. `durations.npz` uses all 7 games. `fold_dur_<gi>.npz` leaves game gi out
   and is loaded only through `dfl_folds.resolve()`, so every DFL game is decoded with a table
   that never saw it.
5. **Post-processing** (LDE's, unchanged). D end -> nearest long ball-gap end within 1 s; alive->D
   start -> first ball-out frame within 6 s; H<->A change -> hand-over-detector peak within
   0.5 s; D before each half's kick-off tag (match only). Measured effect: +0.000 to +0.001.
6. **predict_proba**: the posterior marginals of the segmental forward-backward, with every score
   multiplied by post_tau = 0.2. That is a tempered posterior; 0.2 minimises DFL LOGO frame
   log-loss (0.202, argmax accuracy 0.926). For comparison, LDE's own frame probabilities give
   0.187 / 0.926. The marginals are constant within a lattice cell (<= 0.5 s). predict and
   predict_proba share one cached computation per game (keyed by a hash of the ball array), and
   both are deterministic.

Implementation: the DP is plain numpy (no torch). DP plus marginals take about 6 s per DFL game,
and the LightGBM evidence takes about 12-18 s.

## References

- Ostendorf, Digalakis, Kimball (1996), "From HMMs to segment models", IEEE Trans. Speech & Audio
  Processing — segment scores from frame sums plus explicit duration.
- Yu, S.-Z. (2010), "Hidden semi-Markov models", Artificial Intelligence 174 — explicit-duration
  HSMM Viterbi and forward-backward (the recursions in `semimarkov` follow its
  segment-entry / segment-exit formulation).
- Sarawagi & Cohen (2004), "Semi-Markov conditional random fields" — segmental decoding with
  boundary features (the gamma term).
- LDE and the DFL label statistics: `candidates/learned_dfl_evidence/pipeline.py` docstring.

## Data and licences

- DFL / IDSSE open data: Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of
  spatiotemporal and event data in elite soccer", Scientific Data 12, figshare 28196177,
  https://doi.org/10.6084/m9.figshare.28196177, CC BY 4.0. Fetched by
  `learned_dfl_evidence/fetch_idsse.py`. This candidate only reads the parsed per-frame
  status / possession / half arrays for the duration tables. The LightGBM models are LDE's.
- Metrica Sports sample data (bench dev / holdout), used only through bench.py.
- Nothing is trained on the competition match. There are no hand labels and no ground truth was
  read. evaluation/ was not touched.

## Variants (DFL LOGO spellF1 / macroF1, possessions pred of 3061)

The harness in `tuning/` (`devrun.py` caches the fold-model evidence of the 7 bench --dfl games;
`sweep.py` decodes them with each game's fold duration table) reproduces bench --dfl exactly: the
final config scores 0.6422 in both. Unless stated otherwise, each row changes one thing from the
row it follows.

| step | DFL spellF1 | note |
|---|---|---|
| v0: tau 0.15, no bonus, grid 1 s + touches, 4 states, gamma 0 | 0.5943 / 0.9117 (1864) | |
| tau x bonus_poss grid (0.15-0.6 x 0-4): tau 0.3, bonus 3 | 0.6219 / 0.9148 | plateau 0.621-0.622 |
| w_dur 0 / 0.5 / 1 / 1.5 / 2 | 0.015 / 0.472 / 0.621 / 0.616 / 0.608 | the learned pmf at its natural weight 1 is best |
| gamma 0.1 / **0.3** / 0.6 | 0.6227 / **0.6266** / 0.6250 | kept 0.3 |
| lattice: grid only 0.2 / 0.5 / 1 / 2 s | 0.6261 / 0.6212 / 0.6092 / 0.5761 | |
| lattice: touches + grid 0.2 / **0.5** / 1 / 2 s | 0.6279 / **0.6286** / 0.6266 / 0.6259 | touches recover a coarse grid's loss (2 s: +0.050) |
| post-processing off / refine on | 0.6203 / ±0.001 | kept LDE's post chain |
| ctx=1 (10-state context durations) | 0.6263 (best of a 3x3 tau x bonus grid) | not kept |
| prior_k 0.5 / 1, bonus_dead ±2, w_trans 0.5 / 2, pc_thr 0.02 / 0.15, merge 0.5 s, min_sep 1 / 4 | 0.6248-0.6290 | flat, not kept |
| **ablation**: geometric durations, same mean (= frame HMM), best bonus | 0.6270 vs 0.6284 | explicit durations +0.001 |
| contrast gate: LDE logit (1.25 / 1.25) -> **beta 1.0, gate 0.25**, bonus 1 | **0.6409** / 0.9161 (2404) | plateau 0.639-0.642 for beta 1-1.25, gate 0-0.25 |
| contrast_hw 50 / 200 | 0.6221 / 0.6303 | kept 100 |
| tau 0.2 / 0.4 (bonus 0.5-1.5) | 0.6340-0.6407 | kept 0.3 |
| ctx=1 at the new evidence | 0.6392-0.6416 | noise, not kept |
| **ablation**: geometric durations at the new evidence | 0.6405 (bonus 0) vs 0.6409 | **no measurable value from explicit durations** |
| delta (dead-logit jump on alive<->D boundaries) 0.1 / 0.25 / 0.5 | 0.6410 / 0.6411 / 0.6407 | flat, not kept |
| **gamma 0.6** / 0.8 / 1.0 / 1.5 | **0.6422** / 0.6422 / 0.6409 / 0.5996 | kept 0.6 |
| final checks: grid 0.25 / 1 s, ctx 1 + grid 1 s, post off, touches off (grid 0.2 / 0.5) | 0.6407 / 0.6417, 0.6426, 0.6419, 0.6408 / 0.6370 | all noise; config unchanged |
| post_tau (marginals; DFL frame log-loss) 0.15 / **0.2** / 0.25 / 0.35 | 0.219 / **0.202** / 0.209 / 0.252 | decoding unaffected |

Reference, LDE's decoder on the same cached evidence (`tuning/lde_ref.py`): its own config gives
0.6243 (it reproduces the LDE bench). With beta 1.0 / gate 0.25 it gives 0.6341; with gate 0.5,
0.6347. On Metrica dev it goes from 0.6725 to 0.6767, against 0.6751 for this candidate.

## What the numbers say

- On the same evidence, the segmental lattice decoder beats LDE's Viterbi + post chain by about
  +0.004 (0.6284 vs 0.6243 with LDE's logit) to +0.008 (0.6422 vs 0.6341 with the sharper logit)
  on DFL. The sharper contrast logit adds about +0.010 to either decoder, and it is an evidence
  fix that LDE could adopt too. Total over LDE: +0.018 DFL, +0.003 dev, -0.001 holdout. All of
  these are inside the noise band.
- Duration modelling, which was the hypothesis, does not explain the gain. The duration term is
  needed as a per-segment cost: without it the decoder shatters into ~50k spells. But a geometric
  pmf with the same mean does as well. Context-dependent durations are real in the labels (a
  possession entered from a restart is < 1 s 1.8% of the time; one ending in a stoppage, 25.7%),
  yet they do not help decoding.
- The touch lattice is a good boundary set. With a 2 s grid, adding touches recovers +0.050, and
  touches + 0.5 s grid is ≈ a dense 0.2 s grid.
- Under-splitting remains. Recall at IoU 0.5 by true spell duration (old base, `tuning/diag.py`):
  < 1 s 0.07, 1-2 s 0.30, 2-4 s 0.49, 4-8 s 0.71, 8-16 s 0.85, > 16 s 0.96. Predicted spells
  < 1 s are right only 25% of the time. Adding spells through the bonus stops paying at about
  2400-2500 predicted spells. The limit is the evidence, not the decoder.
- Diversity for the ensemble: predict_proba are segment-posterior marginals, not frame
  classifier outputs. They are smoother and piecewise constant per lattice cell, and slightly
  worse in log-loss than LDE's frame probabilities (0.202 vs 0.187).

## Rebuild

```bash
S=<scratch>; A=$S/arch/semimarkov_lattice
# 1. LDE models: copied from candidates/learned_dfl_evidence (see its rebuild_models.sh)
cp candidates/learned_dfl_evidence/{features.py,model_d.txt,model_ha.txt,model_c.txt,model_meta.json} candidates/semimarkov_lattice/
# 2. duration / transition tables + LOGO folds (deterministic: durations.npz sha256 7170d9047f70)
cp $S/folds_verified.json $A/folds.json
python -I candidates/semimarkov_lattice/train_durations.py --npz $S/idsse/npz --folds-out $A/folds --folds-json $A/folds.json
# 3. bench (OMP_NUM_THREADS=2)
python candidates/bench.py candidates/semimarkov_lattice/pipeline.py --dfl --folds $A/folds.json --save-probs $S/probs/semimarkov_lattice
python candidates/bench.py candidates/semimarkov_lattice/pipeline.py --match --save-probs $S/probs/semimarkov_lattice
python candidates/bench.py candidates/semimarkov_lattice/pipeline.py --holdout --save-probs $S/probs/semimarkov_lattice
```

Folds JSON: `<scratch>/arch/semimarkov_lattice/folds.json`. It holds the five folds_verified.json
entries plus `7170d9047f70 -> [<scratch>/arch/semimarkov_lattice/folds, "dur"]`.
