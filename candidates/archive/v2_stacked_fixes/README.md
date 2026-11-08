# v2_stacked_fixes

Frozen learned DFL frame evidence (team, dead, hand-over) + v1's 3-state Viterbi + v2's dead-ball
boundary snaps. Round 0 stacked seven fixes on v1, round 1 moved the team evidence to the learned
DFL model, round 2 moved the dead evidence to it too and added the hand-over peak snap, and round 3
(this round) sharpened the team logit with a local contrast, swapped in the blackout-trained dead
model and softened the long-gap gate. The shipped pipeline is described in the `pipeline.py`
docstring. The version before round 3 is `pipeline_r2.py`; rounds 0 and 1 are in `archive/`.

## Round 3 results

Keep rule as before: a change stays only if dev spellF1 rises by more than 0.003 with macroF1
not falling more than 0.01, or if it fixes a rules, determinism or robustness problem a judge
raised. Every step was also scored on the four `tuning/views.py` views and on `bench.py --dfl`
(7 open DFL games, leave-one-game-out fold models matching the shipped boosters, tracking-flag
labels with the same meaning as the real ground truth).

| step | clean spellF1 / macroF1 | offset | osd | stress2 | DFL-LOGO | decision |
|---|---|---|---|---|---|---|
| R2 shipped | 0.6603 / 0.8120 | 0.6081 | 0.5886 | 0.5329 | 0.5970 | reference |
| + local-contrast team logit (beta 1, ±100 frames, gate 1) | 0.6769 / 0.8129 | 0.6138 | 0.5797 | 0.5253 | **0.6163** | **kept** (+0.017 clean, all 4 rows up, +0.019 DFL) |
| + round-2 (blackout-trained) `model_d` | 0.6778 / 0.8156 | 0.6133 | 0.5834 | 0.5284 | 0.6156 | **kept** (robustness: osd +0.004, stress2 +0.003, macroF1 +0.003) |
| + soft long-gap gate, centre 1.75 m/s, ramp 0.2 | 0.6760 / 0.8154 | 0.6161 | 0.5848 | 0.5328 | 0.6168 | **kept** (robustness: removes the 2.0 m/s knife edge; offset +0.003, stress2 +0.004) |
| + set-piece `tag_dead` only on a missing / still ball | identical bench | | | | | **kept** (match: 3 frames change) |

Final dev RESULT (`bench.py --match`): **spellF1 0.676, macroF1 0.8154**. Rows: 0.7009 / 0.6966 / 0.6536 / 0.6528. Possessions: 294 / 302 / 272 / 269 (GT 348 / 348 / 329 / 329; R2 predicted 271 / 266 / 256 / 258).

Match: 340 possessions (R2: 314), mean 9.8 s, dead share 0.476 (R2: 0.476).
`submission.csv` md5 is `c702d1d4d9e196f9fa99f47aaaf5d84d`. `make_submission.py` writes a byte-identical file (checked).

### What changed in round 3 (kept)

1. **Local-contrast team logit** (measured-performance judge). `lph' = lph + beta * dz(lph - moving_mean(lph, 100))`, where `dz` is a dead zone of width `gate`. It is applied to the clipped learned team log-odds before the ×6 scale.
   - It targets the main measured loss: short opposing touches in which the team logit dips toward 0 without crossing it. Predicted possessions rise from ~263 to ~284 per dev row against ~339 in the labels, and from 314 to 340 on the match.
   - Dev +0.017, with all 4 rows up. DFL-LOGO goes from 0.5970 to 0.6163, which reproduces the judge's figures exactly.
   - Cost: osd -0.009 and stress2 -0.008 on v2's own views (before the next two changes, which win back about half of it). The gate is what keeps it moderate. Ungated or higher-beta settings were not tried here because the judge measured them losing 0.01-0.02 on the stress views.
   - Computing it per graded block instead of over the whole sequence gives 0.6768 (identical within noise). The code supports both (`contrast_per_block`), and the shipped setting is the whole sequence, as the judge measured.
2. **Round-2 dead model** (generalisation-risk, method-soundness and ideas-from-rivals judges). `lde_frozen/model_d.txt` is now learned_dfl_evidence's round-2 model (md5 858ad6d5..., 706 rounds), trained with extra dropouts and whole-stoppage ball blackouts. The round-1 model never saw long blackouts, and the match has 41% ball-missing frames with long stoppage gaps. `model_ha` and `model_c` stay at round 1, because the judge measured the r2 `model_c` at -0.0013 clean.
   - Clean is flat (+0.0009), osd +0.004, stress2 +0.003 and macroF1 +0.003. DFL is flat (0.6156 against 0.6163; the bench's DFL degradation has no blackouts).
   - `MANIFEST.json`, `model_meta.json` (its `d` entry and `datasets`) and `PROVENANCE.md` are updated. A rebuild of `model_d` alone (`train.py --models d` on the cached round-2 dataset) reproduced the file byte for byte.
3. **Soft long-gap gate** (generalisation-risk and method-soundness judges). `long_gated = sigmoid((1.75 - gap mean player speed) / 0.2) * clip((gap - 2 s) / 3 s, 0, 1)` replaces the hard `gap > 3 s and speed < 2.0 m/s` step. `w_long` stays at 4. (`w_long` 3 is identical; 2 costs 0.002.)
   - Clean -0.0018 (noise), offset +0.003, osd +0.0014, stress2 +0.0044, DFL +0.0012, macroF1 flat.
   - Match sensitivity: moving the gate centre from 1.5 to 2.0 m/s now moves the match dead share 0.470 -> 0.478 (soft gate). With the round-1 model and the hard gate the judge measured 0.459 -> 0.476. Most of that reduction comes from the round-2 dead model; the soft gate removes the step.
   - In round 2 the same soft gate on the hand dead model was rejected (0.6474). With the learned dead model it is flat on clean and better on the stress views.
4. **Set-piece `tag_dead` only on a missing or still raw ball** (generalisation-risk judge). The set-piece prior now adds dead evidence only on frames where the raw ball is missing or moves slower than `sp_dead_still_v = 0.5` m/s. The bench has no tags, so it is a no-op there; on the match it changes 3 graded frames. The whole `tag_dead` term is only worth 410 match frames (against `w_tag_dead = 0`), because the learned dead model already marks almost all of those frames D.
5. **Rules and reproducibility** (rules-reproducibility judge):
   - `lde_frozen/rebuild/` makes the model rebuild self-contained. It holds copies of learned_dfl_evidence's `fetch_idsse.py`, `idsse_manifest.json`, `train.py` and `features.py` (all byte-identical), plus `make_dataset.py`, which is identical except that its `ROOT` line finds the repo root from the deeper folder. `rebuild_models.sh` runs fetch, both datasets and training (`ha,c` on round 1, `d` on round 2), then `cmp`s against the frozen files. v2 no longer depends on another candidate's folder.
   - `make_submission.py` now fails before predicting if `V2_SWEEP` or `V2_CFG` is set, so a leftover sweep variable cannot silently produce a different file.
   - `pipeline_r0.py` and `pipeline_r1.py` were moved to `archive/` with a note. Nothing imports them.
   - `.python-version` pins 3.14.6. The tested platform is macOS arm64, and the expected md5 is above.
   - The training data is cited below (**Training data**), not only in `lde_frozen/PROVENANCE.md`.

### Tried and not kept (round 3)

| change | clean | offset | osd | stress2 | note |
|---|---|---|---|---|---|
| contrast gate 1.5 (r1 dead) | 0.6661 | 0.6114 | 0.5854 | 0.5324 | half the clean gain |
| contrast beta 0.5 / 0.75 (gate 1, r1 dead) | 0.6652 / 0.6702 | 0.6095 / 0.6127 | 0.5846 / 0.5811 | 0.5312 / 0.5249 | |
| contrast beta 1.5, gate 1.5 (r1 dead) | 0.6702 | 0.6118 | 0.5834 | 0.5295 | |
| contrast window ±50 frames (r1 dead) | 0.6633 | 0.6063 | 0.5851 | 0.5312 | |
| contrast window ±200 frames (r2 dead, hard gate) | 0.6810 | 0.6132 | 0.5866 | 0.5285 | +0.003 clean: at the noise line, so the judge's ±100 is kept |
| hard gate at 1.75 m/s (r2 dead) | 0.6776 | 0.6131 | 0.5840 | 0.5308 | soft ramp better on offset / stress2 |
| soft gate with `w_long` 3 / 2 | 0.6757 / 0.6742 | 0.6150 / 0.6130 | 0.5847 / 0.5833 | 0.5333 / 0.5314 | |
| round-2 `model_c` too | | | | | not run again; the judge measured -0.0013 clean |

Not attempted this round: the hierarchical H/A re-decode with hier_touch_decoder's stage B and the 3-way frame vote. Both would make v2 depend on another candidate's code, and the judge measured both as within noise on clean. The DFL-trained short-touch head is the remaining lever for under-splitting: predicted possessions are still ~284 against ~339 per dev row.

## Round 2 results (previous round, for reference)

Same keep rule as before: a change stays only if dev spellF1 rises by more than 0.003 with
macroF1 not falling more than 0.01, or if it fixes a rules, determinism or robustness problem a
judge raised. Every step was also scored on:

- `tuning/views.py` — four views of the dev rows: clean; `offset` (AR(1) sd 1.3 m, tau 1 s); `osd`
  (smooth + offset + player dropout); and the out-of-distribution `stress2` (sd 2 m, tau 3 s, plus
  extra 0.5-2 s ball dropouts).
- `tuning/dfl_logo.py` — v2's full `predict` on the 7 open DFL games with leave-one-game-out fold
  models, scored against the games' own tracking flags. These labels mean the same thing as the
  real ground truth. Two degradations: `r1` (the training distribution) and `r2` (extra dropouts
  plus whole-stoppage ball blackouts, closer to the match's long gaps).

| step | clean spellF1 / macroF1 | offset | osd | stress2 | DFL-LOGO r1 / r2 | decision |
|---|---|---|---|---|---|---|
| R1 shipped | 0.6468 / 0.8152 | 0.6050 | 0.5855 | 0.5476 | | reference |
| + hand-over peak snap 0.5 s | 0.6513 / 0.8169 | 0.6046 | 0.5864 | 0.5460 | 0.4354 / 0.4440 | **kept** (+0.0045, all 4 rows up) |
| + learned dead logit | 0.6602 / 0.8120 | 0.6088 | 0.5884 | 0.5331 | **0.5447 / 0.5395** | **kept** (+0.009 clean, +0.10 DFL) |
| + one merge only (no `pp.merge_short`) | 0.6602 / 0.8120 | 0.6081 | 0.5886 | 0.5329 | | **kept** (judge cleanup, clean identical) |
| + inert code removed, sp_near_m 3 m fixed | identical bench predictions, byte-identical match file | | | | | **kept** |

Final dev RESULT (`bench.py --match`): **spellF1 0.6603, macroF1 0.812**. Rows: 0.6643 / 0.6736 / 0.6496 / 0.6535. Possessions: 271 / 266 / 256 / 258 (GT 348 / 348 / 329 / 329).

Match: 314 possessions (R1: 303), mean 10.6 s, dead share 0.476 (R1: 0.455).
`submission.csv` md5 is `3744a0966ee397f5ba625a02ad04a363`. `make_submission.py` writes a byte-identical file.

### What changed in round 2 (kept)

1. **Hand-over peak snap.** The ideas-from-rivals judge suggested this graft.
   - `lde_frozen/model_c.txt` is learned_dfl_evidence's round-1 hand-over detector. It was frozen before that candidate retrained in round 2 and matches the frozen `model_meta.json`.
   - `p_change` is averaged with the team-swapped prediction.
   - After the merge, each direct H<->A change moves to the `p_change` peak within ±0.5 s.
   - Clean improved by +0.0045, matching the judge's graft measurement exactly. 0.75 s gave 0.6507 and 1.0 s gave 0.6467.
2. **Learned dead evidence.** The dead logit is now `logit p_dead + 4 long_gated + 4 out + tag_dead`. The hand-built S2 terms (bias, slow, few, static, the summed few+long) are gone.
   - Two judges (generalisation-risk and method-soundness) asked for this.
   - DFL-LOGO decided it. On tracking-flag labels, the hand dead model scores 0.435 / 0.444 and the learned one 0.545 / 0.540. That is on par with learned_dfl_evidence's own decoder (0.541).
   - The bench agrees: +0.009 clean, with all 4 rows up.
   - Cost: stress2 drops 0.546 -> 0.533. The measured-performance judge's argument for keeping the hand model rested on stress2. That is a synthetic out-of-distribution view with event-derived labels, and the hand model loses 0.10 on the label semantics the real ground truth uses. So the learned model wins.
   - The `long_gated` (+4) and `out` (+4) hand terms stay. Both are physically motivated, and on DFL r2 `w_long` 0 / 2 / 4 gave 0.539 / 0.543 / 0.540, which is flat.
3. **One short-segment merge.** The method-soundness judge flagged the double merge. `pp.merge_short` before `merge_short_v2` (same `min_poss_s`) is removed. Clean is identical, and offset / osd / stress2 changed by -0.0007 / +0.0002 / -0.0002.
4. **Inert machinery removed** (method-soundness and generalisation-risk judges): the S1 carry, the p_shape fit and v1 ball/shape team terms, the S5 k scaling, `snap_v1`, prune, `shape_lead`, the gap-start snap, the `tag_fix=0` and `dead_v2=0` branches, and the hand dead terms.
   - The set-piece nearest-player fallback used `2 k` m (3.44 m on the match). It is now a fixed `sp_near_m = 3 m`.
   - Bench predictions are unchanged and the match file is byte-identical (checked with cmp), so none of these affected the output.
5. **Rules and reproducibility fixes** (rules-reproducibility judge):
   - `pipeline._load_lde` now *asserts* the lightgbm version and the full md5 of every frozen file against `lde_frozen/MANIFEST.json`. Before, it only warned.
   - `lde_frozen/PROVENANCE.md` gives the sources, the full md5s and the exact retraining commands.
   - `make_submission.py` is a match-only entry point that does not depend on the bench's scratch sample path.
   - The robustness harness moved into `tuning/` (`views.py`, `dfl_logo.py`). `robust_offset.py`, which pointed into `candidates/_scratch/`, was deleted. No file of this candidate refers to `_scratch` any more.
   - Not done here because it is outside this candidate: the root `.gitignore` should get `candidates/_scratch/`. That folder holds match-derived caches.

### Tried and rejected (round 2)

| change | clean | offset | osd | stress2 | note |
|---|---|---|---|---|---|
| learned p_dead > 0.9 OR-mask on raw-seen runs ≥ 1 s (on the hand dead model) | 0.6484 | 0.5946 | 0.5822 | 0.5346 | worse everywhere; 0.95 / 0.8 / 2 s similar |
| soft long-gap gate sigmoid((1.75-v)/0.2) | 0.6474 | 0.6002 | 0.5807 | 0.5454 | |
| max(few, long) instead of the sum | identical | | | | |
| hand `w_long` 3 / 2 | 0.6481 / 0.6210 | | | | |
| unsaturated "calibrated" team emission (eps 1e-12, clip 4.5), scale w 1/2/3/4/6 | 0.6223/0.6339/0.6485/0.6496/0.6513 | | 0.5894 at w 3 | | w ≥ 6 gives output identical to the saturated one, and clip 3 = clip 4.5. Saturation does not decide anything, so no change |
| gated gap-start snap (fast ball / near a line before the gap) | identical | | | | `snap_gap_start 1`: 0.6508 |
| peak snap 0.75 s / 1.0 s | 0.6507 / 0.6467 | | | | |
| learned dead scale 0.5 / 1.5 | 0.6560 / 0.6517 | | | | DFL r2 1.5: 0.5437 |
| learned dead with `w_long` 2 / 6 | 0.6499 / 0.6565 | | | | DFL 0.5486 / 0.5432 at 2 |
| mix 0.3 / 0.5 of the hand dead logit | 0.6546 / 0.6523 | 0.6067 / 0.6049 | 0.5867 / 0.5849 | 0.5379 / 0.5398 | |
| hand + 0.3 learned (additive) | 0.6521 | 0.5995 | 0.5839 | 0.5419 | |
| `min_poss_s` 0.5 (learned dead) | 0.6585 | 0.5997 | 0.5759 | 0.5262 | fails the judge's ±0.005 stress condition; DFL below |
| `min_poss_s` 0.8 / `min_dead_s` 2 | 0.6608 / 0.6592 | 0.6058 / 0.6073 | 0.5852 / 0.5883 | 0.5324 / 0.5372 | noise |
| no D-end gap snap / no ball-out D-start snap | 0.6587 / 0.6597 | | | 0.5281 / 0.5343 | both kept |

DFL-LOGO on the final configuration (r1 / r2 degradation, tracking-flag labels):

| configuration | DFL-LOGO r1 | DFL-LOGO r2 |
|---|---|---|
| shipped | 0.5446 / macro 0.8877 | 0.5391 / 0.8853 |
| without the peak snap | 0.5422 | 0.5366 |
| `min_poss_s` 0.5 | 0.5405 | 0.5346 |

The peak snap also helps on tracking-flag labels. Shorter possessions do not, even though those labels contain more sub-second spells: the evidence cannot place them (round-2 measured-performance judge). Predicted spells are 2163 against 3061 on DFL, so under-splitting remains the main known gap.

`w_tag_sp` stays at 0.8 (v1's weight). The team logit is `±6·clip(lph, 5)` through an eps-1e-4 emission, so a 0.8 set-piece prior can only flip frames where the learned team logit is close to 0. Smart Tagging's set-piece team is direct information about who takes the restart. None of this can be checked on the bench.

## Run

    python candidates/bench.py candidates/v2_stacked_fixes/pipeline.py [--match]   # dev score (+ CSV)
    python candidates/bench.py candidates/v2_stacked_fixes/pipeline.py --dfl --folds FOLDS.json
                                                                                  # DFL-LOGO (fold models)
    python -I candidates/v2_stacked_fixes/make_submission.py [--out PATH]          # match CSV only
    python candidates/v2_stacked_fixes/tuning/views.py '{}' '{"peak_snap_s": 0}'   # robustness views
    sh candidates/v2_stacked_fixes/lde_frozen/rebuild/rebuild_models.sh [OUT]       # rebuild the models

`FOLDS.json` maps each shipped booster's sha256 prefix to its leave-one-game-out fold models (see the `bench.py` docstring). For the shipped models that is `model_ha` and `model_c` to round-1 folds (`--drop-p 0 --blackout-p 0`) and `model_d` to round-2 folds (make_dataset defaults).

Sweeps through the bench: `V2_SWEEP=1 V2_CFG='{"w_lde":4.0}' python candidates/bench.py ...`. `V2_CFG` is ignored, with a warning, unless `V2_SWEEP=1`, and `make_submission.py` refuses to run while either is set. The defaults in `CFG` are the shipped values.

Dependencies are pinned in `requirements.txt` (numpy 2.5.4, pandas 3.1.1, lightgbm 4.7.0) and `.python-version` (3.14.6). Tested on macOS arm64. The lightgbm version and the frozen-file md5s are asserted at load. The output is deterministic: inference runs with num_threads=1 and there is no RNG.

## Training data

The three frozen LightGBM models (`lde_frozen/`) are trained only on the open DFL / IDSSE dataset:

- Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). "An integrated dataset of spatiotemporal and event data in elite soccer." *Scientific Data* 12. figshare article 28196177, version 1, https://doi.org/10.6084/m9.figshare.28196177. Data (c) DFL Deutsche Fussball Liga / IDSSE, licence CC BY 4.0.
- Used: all 7 Bundesliga matches, with the tracking feed's per-frame BallStatus (dead / alive) and BallPossession (home / away) flags as labels. That is about 1.0M frames, sub-sampled every 5th frame and degraded with 3 seeds to look like automatic tracking.
- Not used: no competition-match labels, nothing from the public 5 minutes, and no Metrica data in training. The Metrica sample games are only the dev bench.
- Download: about 2.6 GB, fetched and md5-checked by `lde_frozen/rebuild/fetch_idsse.py`. Rebuild time is about 10-20 min for the fetch and parse plus about 10 min of training. Exact steps and verification are in `lde_frozen/PROVENANCE.md`.
