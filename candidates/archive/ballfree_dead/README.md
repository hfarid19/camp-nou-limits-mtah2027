# ballfree_dead: a ball-free dead-ball / game-state model

**What it is.** A dead-ball (ball out of play) model that never looks at the ball. It works
only from the 22 players' geometry and motion, plus set-piece configurations. It is fused with
learned_dfl_evidence's (LDE) ball-using dead model and LDE's home/away team model, then decoded
with LDE's Viterbi decoder.

**Verdict, plainly.**
* The ball-free model works. On its own it reaches frame-level dead F1 **0.959** and stoppage
  F1 **0.925** on the 7 DFL games (leave-one-game-out).
* LDE's ball-using dead model is still better on DFL: dead F1 0.976, stoppage F1 0.957.
* Fusing the two logits gives DFL LOGO spellF1 **0.6283**, against LDE's 0.6243. That +0.004 is
  inside the noise band, so it is not a real win.
* The final pipeline is about as good as LDE, not better. Its dead-ball edges are tighter: the
  stoppage-start error falls from 0.94 s to 0.65 s.
* The dead model cannot fix the main weakness, which is too few possessions. With
  ground-truth-perfect dead labels and LDE's team model, DFL spellF1 is only 0.6533. The
  remaining gap is in the home/away hand-overs.

## Method

### Ball-free features (`features_bf.py`)

There are 387 per-frame features. They use the player slots only. Every feature is:

* identity-free: counts, means and order statistics over a team's tracked slots;
* team-swap invariant: per-team values enter as the min and max over the two teams;
* pitch-mirror invariant: positions enter as distances to halfway, the box, a flag, the
  touchline or the goal line.

Because of these invariances the model needs no attack direction and no swap augmentation.

| group | features |
|---|---|
| kinematics | speed mean, median, p90 and max; walking, still and sprinting shares; mean acceleration; velocity coherence (\|mean v\| / mean \|v\|) overall and per team |
| shape | centroid distance from halfway and from the centre line; x and y spread; length; width; gap between the two team centroids |
| kick-off | the two teams split by the halfway line; players in the centre circle; distance of the nearest two players to the spot |
| corner and goal line | players within 5 m of a flag; nearest distance to a flag; players within 2 m of the goal line |
| throw-in | players within 1 m of the touchline; nearest distance to the touchline; players outside the pitch |
| box | players in the busier penalty area, per team and in the six-yard box; players waiting just outside the box (penalty, goal kick); final-third crowding |
| wall and marking | the most same-team players within 1.5 m of one player; median distance to the nearest opponent and nearest teammate; the most isolated player |
| tracking | tracked-player counts (the competition tracking collapses during stoppages) |
| temporal | centred means over 1, 4, 10 and 30 s; past and future means over 2 s and 8 s, plus their difference (a restart is a collective acceleration after a still phase); past and future 5 s and centred 15 s min and max of the mean speed; Lang et al.'s shifted copies (±0.4, 0.8, 2, 4, 20, 40, 80 s) of eight 1 s-smoothed core signals |

The most important features by gain, in order:

1. mean acceleration (1, 4 and 10 s means, plus its past and future 2 s means)
2. players outside the pitch
3. 30 s still and walking shares
4. nearest distance to the touchline
5. 30 s wall feature
6. slowest team's speed

### Models

**`model_bfd.txt` (used in the final pipeline).** A LightGBM binary dead/alive classifier.
* Settings: 31 leaves, learning rate 0.05, feature fraction 0.5, deterministic.
* Rounds: 308, chosen by the mean leave-one-game-out log-loss curve.
* LOGO log-loss: 0.098. For comparison, LDE's ball-using `model_d` scores 0.057.
* Frame accuracy on each held-out game: 0.946 to 0.975. Lang et al. report 0.92 on clean,
  identity-labelled tracking.

**`tcn_bfd.npz` (optional; off in the final configuration).** A non-causal temporal
convolutional network (TCN).
* Input: the 49 per-frame base features at 5 Hz.
* Architecture: 8 residual dilated convolutions (kernel 3, dilations 1 to 128, 32 channels),
  giving a receptive field of ±51 s.
* Trained with torch in `train_tcn.py`. Inference is plain numpy in `tcn_np.py`, which matches
  torch to 3e-6.
* LOGO log-loss: 0.1125 at 200 iterations. After that it overfits, since there are only six
  training games per fold.

### Decoder (`pipeline.py`, `lde_pipeline.py`)

The dead log-odds are `w_bf * logit(p_bf) + w_lde * logit(p_lde_d)`, with w_bf = w_lde = 1.

* Team evidence comes from LDE's `model_ha` and `model_c`. Its contrast logit, 3-state Viterbi,
  short-segment merge, hand-over snap and ball-gap terms are copied unchanged from
  `learned_dfl_evidence/pipeline.py`.
* The LDE model files are byte-identical copies (`lde_model_*.txt`, with the same sha256 as in
  `folds_verified.json`).
* `lde_pipeline.py` adds only `evidence_probs()`. It returns the P(home | alive) and P(dead) that
  `decode()` turns into emissions.

`predict_proba` returns P(H) = ph·(1−pD), P(A) = (1−ph)·(1−pD) and P(D) = pD, the decoder's own
per-frame evidence.

### Training data

* Source: the 7 open DFL/IDSSE Bundesliga games. Bassek, Rein, Weber and Memmert (2025), *An
  integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12;
  figshare 28196177. Data © DFL / IDSSE, licensed **CC BY 4.0**. Fetched by LDE's
  `fetch_idsse.py`.
* Degradation: each game is degraded by LDE's `make_dataset.degrade_dfl` (domain randomisation,
  including stoppage blackouts and player collapse) with seeds **41, 42 and 43**. These are not
  the bench's seed 2027 and not LDE's seeds 11 to 13. The bench's DFL caches are never used for
  training.
* Labels: dead where BallStatus is 0.
* Nothing is trained on Metrica or on the competition match.
* Leave-one-game-out fold models: `fold_bfd_<gi>.txt` and `fold_tcn_<gi>.npz`, where gi is the
  index in the sorted game list. They are in
  `$MTAH_WORK/arch/ballfree_dead/folds`,
  mapped in `.../arch/ballfree_dead/folds.json` (which includes `folds_verified.json`). The TCN
  is loaded through `dfl_folds.resolve`.

### References

* S. Lang, R. Wild, A. Isenko, D. Link (2022). *Predicting the in-game status in soccer with
  machine learning using spatiotemporal player tracking data.* Scientific Reports 12:16291,
  doi:10.1038/s41598-022-19948-1. This is the source of the player-only in/out-of-play idea, the
  ±0.4 to 80 s temporal offsets, and the stoppage-level evaluation (stoppages of at least 2 s,
  IoU ≥ 0.5).
* The set-piece configuration features are hand-designed from the Laws of the Game geometry
  (9.15 m circle, box and six-yard box, flag, touchline).

## Results

Scoring: DFL is bench `--dfl` (leave-one-game-out fold models, seed 2027). Dead-ball metrics come
from `dead_report.py`, which wraps the bench's own `bench_dfl`:
* deadF1: frame-level F1 of D;
* stopF1: stoppages of at least 2 s, one-to-one matched at IoU ≥ 0.5, after Lang et al.;
* start/end MAE: mean absolute offset in seconds of matched stoppage starts and ends.

| variant (DFL LOGO) | spellF1 | macroF1 | deadF1 | stopF1 | start / end MAE |
|---|---|---|---|---|---|
| LDE reference (its own dead model) | 0.6243 | 0.9144 | 0.9762 | 0.9574 | 0.94 / 0.72 s |
| V1 pure ball-free LightGBM, no ball-based dead terms or snaps | 0.6055 | 0.8995 | 0.9590 | 0.9249 | 0.97 / 1.30 s |
| V1 + LDE's ball-gap terms and snaps | 0.6055 | 0.8998 | 0.9595 | | |
| V1 at weight 0.5 / 2.0 | 0.6084 / 0.6049 | | | | |
| V1 logit smoothed (half-width 6 / 12 / 25 / 50 frames) | 0.6067 / 0.6065 / 0.6043 / 0.6018 | | | | |
| V2 pure ball-free TCN only | 0.6000 | 0.8960 | 0.9545 | 0.9279 | |
| **V3 pure ball-free: 0.5 LightGBM + 0.5 TCN** (`probs/ballfree_dead_pure`) | **0.6088** | 0.9007 | 0.9598 | 0.9303 | 1.01 / 1.31 s |
| V4 fusion: ball-free 0.5 + LDE dead 1 | 0.6272 | 0.9165 | 0.9789 | | |
| **V4 fusion: ball-free 1 + LDE dead 1 (FINAL)** | **0.6283** | **0.9157** | **0.9781** | **0.9559** | **0.65 / 0.69 s** |
| V4 with ball-free 0.75 / 1.5 / 2 | 0.6288 / 0.6270 / 0.6259 | | | | |
| V4 with LDE dead 0.5 | 0.6250 | | | | |
| V4 with ball-free bias ±0.5 | 0.6288 / 0.6284 | | | | |
| V4 without the ball-based post-processing | 0.6284 | | | | |
| V4 with dead_cost 4 / 6 | 0.6282 / 0.6282 | | | | |
| V4 with w_gap 0 / 1.5 | 0.6289 / 0.6281 | | | | |
| V4 + TCN (TCN share 0.3 / 0.5 / 1.0) | 0.6273 / 0.6271 / 0.6249 | | | | |
| ceiling: ground-truth dead labels + LDE team model | 0.6533 | | | | |

Notes on the table:
* Rows with blank columns were screened with a cached-evidence re-decoder (scratch
  `sweep.py` / `sweep2.py`). It reproduces the bench exactly on the rows run both ways (0.6055,
  0.6243, 0.6283, 0.6088).
* Everything inside about ±0.003 is a plateau.
* The final setting is the simplest point on that plateau: ball-free weight 1, LDE dead weight 1,
  no TCN, LDE's decoder unchanged.
* Restart analysis: only 5.2% of the 699 restarts had the wrong team for the first possession, so
  set-piece recognition was not also turned into a team prior.

**Final scores** (`pipeline.py` as shipped):

| split | spellF1 | macroF1 | possessions (predicted / true) |
|---|---|---|---|
| DFL LOGO | 0.6283 | 0.9157 | 314/511, 320/500, 295/402, 324/456, 309/421, 342/401, 301/370 |
| Metrica dev | 0.6686 | 0.8049 | 283, 294 / 348 and 271, 266 / 329 |
| Metrica holdout G3 (run once, at the end) | 0.6712 | 0.8421 | 300, 314 / 364 |

LDE scores 0.6243 / 0.6725 / 0.6845 on the same three splits, so this pipeline is +0.004 / −0.004
/ −0.013 against it. All three are within noise.

**The pure ball-free variant** (V3, purely ball-free dead evidence) scores:
* Metrica dev: 0.6184 spellF1, 0.7562 macroF1 (V1: 0.6148, 0.7505).
* Its mean P(D) is 0.48 to 0.53 on Metrica, where the fusion gives 0.43 to 0.46.

So the ball-free model over-calls dead on another tracking provider. Player kinematics and set-up
differ from DFL, which is domain shift. On the competition match its mean P(D) is 0.57, against
0.55 for the fusion.

**Match** (`submission.csv`):
* 317 possessions, dead share 0.482, mean possession 10.4 s.
* Two `--match` runs are byte-identical (cmp).
* The pure-variant match file (scratch only) has 315 possessions and dead share 0.506.

## Rebuild

Run from the repo root, with `S=$MTAH_WORK`.

```bash
python -I candidates/learned_dfl_evidence/fetch_idsse.py --games 7          # DFL npz (LDE's fetcher)
OMP_NUM_THREADS=1 python -I candidates/ballfree_dead/make_dataset.py --npz $S/idsse/npz --out $S/arch/ballfree_dead/ds --jobs 2
OMP_NUM_THREADS=1 python -I candidates/ballfree_dead/train.py --ds $S/arch/ballfree_dead/ds --folds $S/arch/ballfree_dead/folds --jobs 2   # ~10 min
# optional TCN (off in the final config; ~22 min, 2 threads, torch only):
OMP_NUM_THREADS=1 python -I candidates/ballfree_dead/make_seq.py --npz $S/idsse/npz --out $S/arch/ballfree_dead/seq --jobs 2
OMP_NUM_THREADS=2 python -I candidates/ballfree_dead/train_tcn.py --seq $S/arch/ballfree_dead/seq --folds $S/arch/ballfree_dead/folds
# folds.json = folds_verified.json + {"e0462fc2b256": [folds dir, "bfd"], "18741562b777": [folds dir, "tcn"]}
python candidates/bench.py candidates/ballfree_dead/pipeline.py --dfl --folds $S/arch/ballfree_dead/folds.json
python candidates/ballfree_dead/dead_report.py candidates/ballfree_dead/pipeline.py --folds $S/arch/ballfree_dead/folds.json
python candidates/bench.py candidates/ballfree_dead/pipeline.py --match
# pure ball-free view (V3): BFD_SWEEP=1 BFD_W_LDE=0 BFD_BALL_POST=0 BFD_A_TCN=0.5 python candidates/bench.py ...
```

* The LDE model copies (`lde_model_{d,ha,c}.txt`) are rebuilt by LDE's own `rebuild_models.sh`.
* Model hashes are checked at load: `model_bfd.txt` against `model_bfd_meta.json` (sha256
  e0462fc2b256…) and the LDE files against `lde_model_meta.json`.
* Deterministic: LightGBM uses `num_threads=1` at inference, and the numpy TCN is exact.
* `BFD_*` / `BFDL_*` environment overrides are honoured only with `BFD_SWEEP=1` (tuning).
* Runtime is about 23 s per 90-minute game. LDE's features take about 12 s of that and
  features_bf about 7 s.
