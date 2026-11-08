# ball_radar: Ball Radar-style ball / possession inference + stacked LightGBM

The competition ball is missing on ~41% of graded frames. This candidate trains a
Ball Radar-style network (Kim et al., KDD 2023) on the open DFL/IDSSE games to infer, from the
anonymous player tracks and whatever ball fragments exist, (a) the ball position in the gaps,
(b) player-level ball possession and (c) team possession / dead ball. Its outputs are then used
two ways by a LightGBM stage: the LDE feature set is recomputed on the radar-imputed ball, and the
radar's own team / dead / possessor-mass outputs are added as feature columns. Decoding is
learned_dfl_evidence's Viterbi + snapping chain, unchanged.

## Results (all numbers from bench runs in this session)

| | DFL LOGO spellF1 / macroF1 | Metrica dev | Metrica holdout G3 (run once) |
|---|---|---|---|
| learned_dfl_evidence (reference, re-run here) | 0.6243 / 0.9144 | 0.6725 / 0.8145 | 0.6845 |
| **ball_radar (final)** | **0.6465 / 0.9229** | **0.7089 / 0.8256** | **0.7006 / 0.8529** |

DFL per game (spellF1, possessions predicted / true), final vs LDE:

| game | ball_radar | LDE |
|---|---|---|
| J03WMX | 0.5840 (324/511) | 0.5575 (317/511) |
| J03WN1 | 0.5906 (348/500) | 0.5776 (313/500) |
| J03WOH | 0.6894 (303/402) | 0.6617 (295/402) |
| J03WOY | 0.6331 (335/456) | 0.6200 (324/456) |
| J03WPY | 0.6629 (326/421) | 0.6408 (310/421) |
| J03WQQ | 0.6732 (334/401) | 0.6415 (346/401) |
| J03WR9 | 0.6924 (292/370) | 0.6711 (297/370) |

Better than LDE on all 7 games (+0.022 mean). It still under-splits: 292-348 possessions
predicted against 370-511 true, so this is not a fix for the main known weakness.
Match: 332 possessions, dead share 0.469, no X inside the halves. Two `--match` runs produced
byte-identical `submission.csv` (sha256 b1f0c284...aa792) and `match.npy`. Each DFL game takes
about 30 s on one thread, and the match about 60 s.

### Ball-position error (radar fold models on held-out games, metres vs the clean DFL ball)

Bench-like degradation (`tune_on_sample_data.degrade`, seed 41, my own copy, not the bench cache),
7 held-out games, 5 Hz samples:

| frames | radar | linear interpolation of the observed ball |
|---|---|---|
| alive play, ball missing | **2.70** (median 1.36) | 3.17 (median 1.85) |
| ball missing, gap > 2 s | **2.66** (median 1.07) | 2.91 (median 1.20) |
| all ball-missing frames | 1.92 | 2.04 |
| ball seen (radar output vs noisy observation) | 0.55 | 0.25 (the pipeline keeps the observed ball there) |

Team head on alive frames: 85.1% H/A accuracy (J03WR9 only 71.9%). Dead head: 96.8%.
On the harder LDE-style degradation (`make_dataset.degrade_dfl`, seed 51: AR(1) offsets,
low-pass, stoppage blackouts), the out-of-fold alive-gap error is 3.84 m against 3.92 m for linear
interpolation, which is almost no gain. Long blackouts there sit mostly inside stoppages and at
their edges, and the network does not localise the ball much better than interpolation in those
cases. Without any ball (the setting of the original paper), Kim et al. report about 3.7 m.
The radar's value here is mostly the possession evidence, not the ball position (see the ablations).

## Method

1. **Ball Radar network** (`train_radar.py` in torch for training, `radar_common.forward` as an exact
   numpy mirror for inference, parity checked to 1e-7). It runs at 5 Hz on canonical coordinates
   (the home team always attacks +x), so a home/away swap is "negate the coordinates and the team flag".
   - Per-slot input: position, velocity, team flag, offset and distance to the linearly
     interpolated ball, a "ball seen" flag and speed. Per-frame input: the interpolated ball, the
     time since and until the ball was last seen, the gap length, the tracked-player count and speeds.
   - Set encoder: a per-slot MLP, then one multi-head self-attention block over all slots of a
     frame (the Set Transformer SAB: permutation-equivariant, empty slots masked), then masked
     mean/max pooling per team (partially permutation-invariant context, as in the paper).
   - Player possession classifier (the paper's PPC): a slot-wise Bi-GRU over time with shared
     weights, giving a softmax over the slots plus a "dead / no player" class. Target: the slot
     nearest the clean ball when it is within 1 m, otherwise the next such controller in the same
     alive spell (the paper's "next controller" rule). Dead frames go to the extra class.
   - Ball trajectory regressor (the paper's BTR): per-team pooling of [slot embedding, slot GRU
     state, possession prob], the expected possessor position and the team masses, fed to a
     frame Bi-GRU. Its output is the ball position as a residual on the linear interpolation,
     plus a team head over H/A/D (DFL BallPossession / BallStatus flags).
   - Loss: SmoothL1 on the ball, weighted to the frames that need imputing (alive gap 1, dead gap
     0.3, seen 0.05), plus CE(owner) and CE(team). Bi-GRU replaces the paper's Bi-LSTM. The reality
     loss is implemented (`--l_real`) but was not used (no time to test it).
     The paper's rule-based post-processing was not used either.
   - Inference averages each run with the home/away-swapped run, and the 5 Hz outputs are linearly
     interpolated to 25 Hz.
   - Training: 800 iterations, batch 12 windows of 100 steps (20 s), Adam with one-cycle 1e-3,
     111k parameters, about 3.4 min per model on one CPU thread.
2. **Stacked LightGBM** (`features_br.py`, `feat_data.py`, `train_lgb.py`): the LDE feature set
   recomputed on the radar-imputed ball. Gap structure (ball_seen, since/until, last confident
   control) still comes from the raw ball. On top come 13 radar columns: the team and dead log-odds
   (with 0.5 s / 2 s means and a past-minus-future contrast), the dead-class mass, the possession
   mass per team and an imputed flag. There are three binary boosters (dead, home-vs-away with
   team-swap augmentation, and a ±4-frame hand-over detector), lr 0.08 with rounds picked by the
   LOGO log-loss curve (d 147, ha 252, c 135).
3. **Decoder**: learned_dfl_evidence's `decode` (copied as `lde_core.py`) with its default constants:
   contrast team logit, Viterbi with switch 0.75 / dead 5, merge_short_v2, gap-end D snaps, and
   hand-over peak snapping, which here uses the stacked hand-over model. No LDE booster is used in the final
   configuration. `predict_proba` returns the stacked P(H), P(A), P(D) (rows sum to 1;
   P(D) = p_dead, P(H) = (1 - p_dead) p_home).

### Leave-one-game-out discipline

- Radar fold k is trained on the 6 other games (`--exclude k`). The full model is trained on all 7.
- LightGBM fold k is trained only on games j != k, with features from **radar fold k** (in-sample
  on those games, the same way the full LightGBM sees full-radar features on all games). Game k's
  features from radar fold k are out-of-sample and are used only for the LOGO curve that picks the
  rounds, the same procedure LDE used. So no artefact used to score game k has seen game k.
  The in-sample radar error on the training games was barely below the out-of-sample error
  (for example 4.20 vs 4.18 m), so the stacking does not over-trust in-sample radar outputs.
- `--folds` map: `$MTAH_WORK/arch/ball_radar/folds.json`.
  `radar.npz` is resolved through `dfl_folds.resolve` (kind `radar`). The bench swaps in
  `model_brd/brha/brc.txt` (kinds `brd`, `brha`, `brc`). The file also carries the
  folds_verified.json entries, which the final configuration does not need.
- Degradation seeds: radar training 31, 32, 33 (`degrade_dfl`); LightGBM features 51; ball-error
  evaluation 41 (bench-like). None of them is a bench seed (2027 / 21), and the bench caches were never read.

## Variants tried (DFL LOGO spellF1, bench --dfl seed 2027)

| step | DFL spellF1 / macroF1 | kept |
|---|---|---|
| LDE reference | 0.6243 / 0.9144 | - |
| radar v0: absolute ball regression, held-out ball error worse than linear interpolation in gaps (gap 3.00 vs 1.95 m, game 0) | not benched | no |
| radar v1: residual on the interpolated ball (gap alive 2.55 vs 2.98 m, game 0); fusion with LDE w 0.3/0.3 on game 0 only: 0.5667 vs 0.5575 | game 0 only | replaced by v2 |
| radar v2: gap-weighted ball loss (game-0 gap error unchanged, cheaper at 800 iterations) | - | yes |
| A. LDE + radar team/dead log-odds, w_h = w_d = 0.3 | 0.6260 / 0.9166 | no (noise) |
| A. w_h = w_d = 0.6 | 0.6256 / 0.9173 | no |
| A. w_h 0.3 only / w_d 0.3 only | 0.6242 / 0.6265 | no |
| A. radar heads alone (no LDE) | 0.5203 / 0.8920 (244-268 poss) | no, but a different view |
| B. stacked LightGBM d/ha (imputed ball + radar columns), LDE's c | **0.6462 / 0.9229** | yes |
| B. 50/50 blend with LDE logits | 0.6407 / 0.9222 | no |
| B. 25/75 blend LDE/stacked | 0.6470 / 0.9235 | no (+0.0008, noise; adds LDE dependence) |
| B. decoder sweeps: switch 0.5 / 1.0, dead 4 / 6, min_poss 0.75 | 0.6470 / 0.6450 / 0.6461 / 0.6461 / 0.6470 | no (flat) |
| B. no local-contrast team logit | 0.6286 | no |
| B. + stacked hand-over model c, LDE-free (final) | **0.6465 / 0.9229** | yes (same score, no LDE booster at all) |
| ablation: same LightGBM recipe and data, **no radar at all** | 0.6089 / 0.9072 | - |
| ablation: imputed ball features, **no radar columns** | 0.6209 / 0.9163 | - |

Attribution: against the same recipe without the radar, the imputed ball adds about +0.012 and
the radar possession columns add about +0.026 more. The LightGBM recipe alone (one degradation
seed, fewer rounds) is 0.015 below LDE, so the gain over LDE comes from the radar.

Not tried for lack of the ~40 min training budget: a second LightGBM degradation seed (LDE uses
3), the reality loss, the paper's rule-based post-processing, longer radar training.

## Data and licences

- DFL / IDSSE open data: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). "An integrated
  dataset of spatiotemporal and event data in elite soccer." Scientific Data 12.
  https://doi.org/10.6084/m9.figshare.28196177, data (c) DFL / IDSSE, CC BY 4.0. Fetched by
  `learned_dfl_evidence/fetch_idsse.py` (cached outside the repo, never committed). Only the
  tracking (players, ball, BallStatus, BallPossession) of the 7 games is used.
- No Metrica data and no competition-match data are used for training. There are no pre-trained
  weights and no hand labels.
- Reference: Kim, H., Choi, H.-J., Kim, C. J., Yoon, J., Ko, S.-K. (2023). "Ball Trajectory
  Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM."
  KDD 2023, arXiv:2306.08206. Set Transformer: Lee et al., ICML 2019.

## Files

`pipeline.py` (predict / predict_proba), `radar_common.py` (inputs + numpy network),
`features_br.py` (LDE features with ball_imp / radar columns), `features.py` + `lde_core.py`
(copies of learned_dfl_evidence's features.py / pipeline.py, used for the decoder), `prep_data.py`,
`train_radar.py`, `feat_data.py`, `train_lgb.py`, `eval_ball.py`; models `radar.npz` (447 KB),
`model_brd.txt`, `model_brha.txt`, `model_brc.txt`, `model_br_meta.json` (sha256 of every model,
including radar, asserted at load; lightgbm version asserted).
`BR_SWEEP=1` enables the `BR_*` overrides used for the variant table. They are ignored otherwise.

## Rebuild (python = the scratchpad venv; A = <scratchpad>/arch/ball_radar; C = candidates/ball_radar)

```
python -I $C/prep_data.py --out $A/seq --seeds 31 32 33          # radar training sequences
python -I $C/prep_data.py --out $A/seq --seeds 41 --eval          # ball-error diagnostic only
for k in 0 1 2 3 4 5 6; do
  OMP_NUM_THREADS=1 python -I $C/train_radar.py --seq $A/seq --exclude $k \
      --out $A/folds/fold_radar_$k.npz --iters 800 --threads 1    # (fold 0 was run with --threads 2)
done
OMP_NUM_THREADS=1 python -I $C/train_radar.py --seq $A/seq --out $A/radar_full.npz --iters 800 --threads 1
cp $A/radar_full.npz $C/radar.npz
python -I $C/feat_data.py --out $A/feat --radar 0 1 2 3 4 5 6 full
python -I $C/train_lgb.py --feat $A/feat --folds-out $A/folds --out $C --models d,ha --radar $C/radar.npz
python -I $C/train_lgb.py --feat $A/feat --folds-out $A/folds --out $C --models c --radar $C/radar.npz
python -I $C/eval_ball.py --seq $A/seq --folds $A/folds          # ball error table
# folds.json: sha256[:12] of radar.npz / model_brd / model_brha / model_brc -> [$A/folds, radar|brd|brha|brc]
python candidates/bench.py candidates/ball_radar/pipeline.py --dfl --folds $A/folds.json
```

Torch training is not bit-reproducible across thread counts, so a retrain changes the model
sha256s and model_br_meta.json. Inference is deterministic (numpy float64, LightGBM
num_threads=1). Wall time for the final recipe: radar 8 x ~3.4 min (two 1-thread streams,
~15 min), features 56 radar runs (~17 min, two streams), LightGBM ~4 min. About 25 more minutes
went into the discarded radar v1 runs.
