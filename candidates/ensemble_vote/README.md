# ensemble_vote: a majority vote over the possession candidates

The user asked for an ensemble that "checks the vote from each architecture on the flag and takes the majority vote". This folder builds that vote over all 11 candidates and a soft-vote alternative. The member set and knobs were chosen on DFL only, with nested leave-one-game-out (LOGO).

**Shipped (`pipeline.py`): a hard majority vote of temporal_tcn, handover_mstcn and ball_radar, followed by merge_short smoothing (1.0 s).**

| | DFL LOGO spellF1 / macroF1 | Metrica dev | Metrica holdout G3 |
|---|---|---|---|
| **ensemble, hard vote (shipped)** | **0.6619** / 0.9268 | 0.7069 | **0.7045** |
| ensemble, soft vote (`pipeline_soft.py`) | 0.6604 / 0.9268 | 0.7088 | 0.7018 |
| temporal_tcn (best single on DFL) | 0.6544 / 0.9230 | 0.6837 | 0.6793 |
| ball_radar (best single on dev / holdout) | 0.6465 / 0.9229 | **0.7089** | 0.7006 |
| handover_mstcn | 0.6523 / 0.9202 | 0.6877 | 0.6764 |
| learned_dfl_evidence (previous best) | 0.6243 / 0.9144 | 0.6725 | 0.6845 |

These numbers are live bench runs of these pipelines. The DFL figure for the ensemble is already the honest nested estimate. Under `bench --dfl`, game *gi* uses `fold_cfghard_<gi>.json`, the rule chosen on the other 6 games only, and every member model is its LOGO fold model. The offline nested computation gave the same values (0.6619 and 0.6605). Selecting on all 7 games gives in-sample scores of 0.6633 for the hard vote and 0.6672 for the soft vote.

## Is it worth submitting? Yes, with modest expectations

- **DFL:** the hard vote scores +0.0075 over the best single member in hindsight (temporal_tcn). Against an honest "pick the best member on the 6 training games" baseline (0.6491 nested) it scores +0.013. It beats temporal_tcn on 5 of 7 games.
- **Metrica dev:** 0.7069, the same as ball_radar's 0.7089 (and ball_radar is a member).
- **Holdout:** 0.7045 is the best of every candidate, +0.004 over ball_radar and +0.020 over LDE.
- **Noise:** every one of these gaps is inside the ~0.04 spellF1 noise band. It is the only entry that is at or near the top on all three benches. Each single member wins at most one bench: TCN wins DFL, radar wins dev and holdout.
- **Under-splitting is not fixed.** DFL possessions are 2334 predicted against 3061 true, compared with TCN's 2467 and LDE's 2202. The match gives 335 possessions.

## What the vote is

Shared code is in `ensemble_core.py`.

**(a) Hard majority vote (`pipeline.py` = `pipeline_hard.py`).**
- Every member's own `predict` labels each frame H, A or D. Each member has one vote, and the plurality label wins.
- Fixed tie-break: each vote also carries a bonus of 1e-3 × (K − rank), where members are ranked by DFL LOGO spellF1. A tie therefore goes to the label of the best-ranked member among the tied voters. With 3 voters a tie is a 1-1-1 split, which goes to temporal_tcn.
- Optional smoothing: LDE's `merge_short_v2`. It merges segments shorter than `min_poss_s` / `min_dead_s` into their neighbours, shortest first.
- `predict_proba` returns the vote shares (fraction of votes for H, A and D).

**(b) Soft vote (`pipeline_soft.py`).**
- A weighted pool of the members' P(H, A, D). Two pooling rules were tried:
  - arithmetic: Σ w P
  - geometric / log-linear: exp Σ w log P, renormalised
- Then one decode with learned_dfl_evidence's `decode()`: its 3-state Viterbi and costs, gap-end and out snaps, `merge_short_v2`, hand-over peak snap on LDE's `model_c`, and kick-off D. The decode takes p_dead = P(D) and p_home = P(H) / (P(H) + P(A)).
- Decoder variants:
  - c0: members' evidence as given
  - c1: plus LDE's local-contrast team logit
  - c2: plus LDE's long-gap dead term
- `predict_proba` returns the pooled P.

### Running members live in one process (`members.py`)

- Every candidate folder has modules with the same names (`features`, `lde_core`, `tcn_np`, ...). Each member's `pipeline.py` is therefore executed under a private name, with its own folder first on `sys.path` and clashing names removed from `sys.modules` before and after.
- Members are re-loaded every time the ensemble module is executed. The bench re-executes it for each DFL game, so the members' lazily cached boosters are always that game's fold models.
- `sys.dont_write_bytecode` keeps `__pycache__` out of the other candidates' folders.
- No member needs torch at inference, so there is no libomp clash.
- Verification: the members' probabilities obtained this way are byte-identical to the members' own saved `probs/` files, for all files compared. Every member's bench score was reproduced exactly (see "Members" below).

### Wrappers for the three candidates without `predict_proba`

All three wrappers are in `members.py` and share a single feature pass with `predict`:

| candidate | P(D) | P(H) |
|---|---|---|
| learned_dfl_evidence | p_dead, after the decoder's own long-gap term | (1 − pD) × p_home, after the decoder's own contrast term; these are the exact inputs of LDE's Viterbi |
| v2_stacked_fixes | exp(E / temperature) of its `build_evidence` | same |
| hier_touch_decoder | sigmoid of its stage-A dead logit | (1 − pD) × sigmoid of its stage-B base team logit, after its contrast |

## Members (DFL LOGO, reproduced live through `members.py`)

| member | DFL spellF1 | DFL possessions (true 3061) | dev | status |
|---|---|---|---|---|
| temporal_tcn | 0.6544 | 2467 | 0.6837 | working |
| handover_mstcn | 0.6523 | 2485 | 0.6877 | working |
| ball_radar | 0.6465 | 2262 | 0.7089 | working |
| semimarkov_lattice | 0.6422 | 2417 | 0.6751 | working |
| pitch_control | 0.6288 | 2217 | 0.6784 | working |
| ballfree_dead | 0.6283 | 2205 | 0.6686 | working |
| pathcrf_dynamic | 0.6248 | 2232 | 0.6649 | working |
| learned_dfl_evidence | 0.6243 | 2202 | 0.6725 | working (wrapped) |
| skillcorner_domain | 0.6185 | 2143 | 0.6850 | partial (used as a candidate member) |
| v2_stacked_fixes | 0.6168 | 2238 | 0.6760 | working (wrapped) |
| hier_touch_decoder | 0.5987 | 2229 | 0.6632 | working (wrapped) |

- No member failed or had broken probabilities, so none was excluded up front.
- `ballfree_dead_pure` and pitch_control's PC-only variant exist only as saved probabilities, not as pipelines. They could not run live, so they were not used.

### Frame-error correlation (phi of the per-frame error indicators, pooled over DFL)

Seven members are essentially learned_dfl_evidence with a different decoder or one model swapped. Their errors correlate with LDE's at 0.81-0.91. Only the two TCNs (0.63-0.66 against LDE, and 0.74 with each other) and ball_radar (0.71) disagree in a useful way. That is exactly the set the selection picks. Full matrix: `logs/diversity_dfl.txt` in the scratch folder.

## Every variant (DFL, 7 games, spellF1)

### Hard vote, fixed member sets

Members are ordered by DFL score. These rows are in-sample: they are not selected per fold.

| set | no smoothing | merge 0.5 s | merge 1.0 s |
|---|---|---|---|
| all 11 (the literal "every architecture votes") | 0.6299 | 0.6341 | 0.6313 |
| top 3 | 0.6574 | 0.6623 | 0.6633 |
| top 5 | 0.6553 | 0.6606 | 0.6599 |
| top 7 | 0.6411 | 0.6453 | 0.6456 |

**Voting with all 11 is worse than the best single member** (0.630 vs 0.654). The 7 LDE-like members outvote the different ones, and the vote reproduces LDE with slightly fewer errors. The same happens on dev: all 11 gives 0.6791 against 0.7069 for the top 3.

### Hard vote, nested LOGO

Within each fold the search covers top-k (k = 3, 5, 7, 9, 11) and a stepwise add/remove search from the top 3, each with every smoothing option.

- Without smoothing: **0.6541**.
- With smoothing selected (m05 or m10, or none allowed): **0.6619**. The live bench gives the same 0.6619.
- Choices per fold: top 3 with m10 in 5 folds, top 3 with m05 in 1 fold, and in 1 fold the stepwise set {handover_mstcn, temporal_tcn, ball_radar, pitch_control} with m10.
- Final rule on all 7 games: top 3 (temporal_tcn, handover_mstcn, ball_radar) with m10.

### Soft vote, fixed sets with equal weights

In-sample, arithmetic / geometric pooling:

| set | c0 | c1 | c2 |
|---|---|---|---|
| all 11 | 0.6370 / 0.6384 | 0.6440 / 0.6461 | 0.6398 / 0.6443 |
| top 3 | 0.6503 / 0.6498 | 0.6649 / 0.6651 | 0.6614 / 0.6621 |
| temporal_tcn + ball_radar | 0.6526 / 0.6516 | 0.6668 / 0.6674 | 0.6629 / 0.6642 |
| temporal_tcn alone, re-decoded | 0.6441 / 0.6441 | 0.6585 / 0.6593 | 0.6557 / 0.6565 |

Part of the soft vote's apparent gain is the decoder, not the pooling: TCN alone through this decoder with c1 (no LDE long-gap term) already scores 0.659 against its own 0.6544.

### Soft vote, nested LOGO

- Each fold searches pooling (arith, geo) × weights (equal, ∝ DFL score) × decoder (c0, c1, c2) on the top 3 and on all members. It then tries top-k and a stepwise member search with the best combination.
- Nested score: **0.6605**. The live bench gives 0.6604.
- Folds chose geometric or arithmetic pooling, always score weights and c1, with TCN + radar (+ semimarkov_lattice in 3 folds, + handover_mstcn in 1).
- Final rule on all 7 games: TCN + radar, geometric, weights ∝ score, c1.
- Its in-sample score (0.6672) overstates it more than the hard vote's does (0.6633).

### Choice

The hard vote was chosen **before** any holdout run because:
- its nested DFL score is higher (0.6619 vs 0.6605, a tie);
- it has fewer knobs (one smoothing switch, against pooling, weights, decoder and set);
- it is what the user asked for.

The holdout was then run once for each of the two pipelines.

## Rules and data

- No training happens here. The only fitted quantities are the member subset, the smoothing switch and, for the soft vote, the pooling / weight / decoder choice. They were fitted on DFL member outputs from the members' own LOGO fold models.
- DFL labels are recomputed from the raw IDSSE flags exactly as `bench.dfl_prepared` does. The bench caches were never read.
- Metrica dev was used only for reporting. The holdout was run once per pipeline, at the end, after the shipped choice was fixed. `evaluation/gt` and `evaluation/reports` were never opened, and `run_evaluation.py` was never run.
- The rule files were chosen with DFL labels, so they go through `dfl_folds.resolve()`. `config_hard.json` → `fold_cfghard_<gi>.json` and `config_soft.json` → `fold_cfgsoft_<gi>.json`, both in scratch `arch/ensemble_vote/folds_cfg/`. The merged folds JSON at scratch `arch/ensemble_vote/folds.json` contains every member's entries (including `folds_verified.json`) plus these two.
- Data, inherited from the members:
  - DFL/IDSSE open data (Bassek et al. 2025, figshare 28196177, CC BY 4.0).
  - skillcorner_domain's model_d also uses SkillCorner open data (MIT), but that member is not in the shipped vote.
- Determinism:
  - Two `--match` runs gave byte-identical `submission.csv` (sha256 5dab22df…c6c6) and `match.npy`.
  - Match: 335 possessions, mean 10.1 s, dead share 0.470, 0 X inside the halves. `validate_submission.py` reports Valid.
- Cost:
  - The hard vote runs 3 members: about 20 s per DFL game and about 27 s per Metrica game on 1 thread.
  - The match peaks at ball_radar's memory use (about 6 GB).
- `submission.csv` is a match label file. Do not commit it publicly.

## Files and rebuild

Code:
- `pipeline.py` (shipped, hard vote), `pipeline_hard.py` (identical rule), `pipeline_soft.py`
- `vote_pipeline.py`, `ensemble_core.py`, `members.py`
- `config_hard.json`, `config_soft.json`
- `pipeline_member.py`: a single-member collector, used for offline fitting only
- `tuning/fitlib.py`, `tuning/fit.py`, `tuning/write_configs.py`, `tuning/diversity.py`

Logs: everything is in scratch `arch/ensemble_vote/logs/`.

```bash
S=$MTAH_WORK
E=$S/arch/ensemble_vote
# 1. collect each member's live codes / probs / LDE side info (bench, 1 thread, 2 streams)
$E/collect.sh dfl <member> ...      # uses pipeline_member.py, writes $E/collect/<member>/
$E/collect.sh dev <member> ...
# 2. nested fits and rule files
cd candidates/ensemble_vote/tuning
$S/venv/bin/python fit.py hard && $S/venv/bin/python fit.py soft
$S/venv/bin/python write_configs.py hard 2 && $S/venv/bin/python write_configs.py soft 0
# 3. benches (from the repo root, OMP_NUM_THREADS=1)
$S/venv/bin/python candidates/bench.py candidates/ensemble_vote/pipeline.py --dfl --folds $E/folds.json --save-probs $S/probs/ensemble_vote
$S/venv/bin/python candidates/bench.py candidates/ensemble_vote/pipeline.py --match --save-probs $S/probs/ensemble_vote
$S/venv/bin/python candidates/bench.py candidates/ensemble_vote/pipeline.py --holdout --save-probs $S/probs/ensemble_vote
```
