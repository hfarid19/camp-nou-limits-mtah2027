# hier_touch_decoder

Hierarchical two-stage possession decoder for the bench interface (`predict(...)` in `pipeline.py`).

1. **Stage A, alive/dead.** A 2-label minimum-duration Viterbi (D at least 2 s, alive at least 1 s, cost 5)
   on a dead-ball logit. Since round 3 this has v2's form: the learned DFL dead log-odds + 4 x ball out
   + 4 x long ball gap (soft player-speed gate, sigmoid((1.75 - speed) / 0.2)) + corrected Smart Tagging
   dead windows. Then the D edges are snapped: the D end goes to the end of a long ball gap, and the D start
   goes back to the first ball-out frame (or to a long-gap start). Kick-off pre-D comes from a
   `kick off (start)` tag whose restart falls within 180 s of the half start.
2. **Stage B, team inside each alive run** (run edges are hard boundaries). A 2-label minimum-duration
   Viterbi (dmin 25 frames, relaxed for the first and last segment). The evidence is 4 x the learned team
   log-odds l, sharpened since round 3 by a dead-zoned local contrast
   l + 0.75 * sign(c) * max(|c| - 1.5, 0) with c = l - moving_mean(l, +-100 frames), plus a small Smart
   Tagging phase prior (weight 0.2). The switch cost (4) drops at a team-velocity reversal.
3. **Post-processing.** Team-change snap to the new team's first control frame within 0.5 s (radius 1.5 m,
   no longer scaled by the match k), and a merge of spells under 0.5 s.

The learned evidence is a **frozen snapshot** in `lde_frozen/` of `learned_dfl_evidence`'s `features.py`,
`model_d.txt`, `model_ha.txt`, `model_c.txt` and `model_meta.json`, identical to
`v2_stacked_fixes/lde_frozen/`. The md5 of every file and the lightgbm version are listed in
`lde_frozen/MANIFEST.json` and asserted at load.

## Training data

The LightGBM models are trained only on the open DFL / IDSSE dataset: 7 Bundesliga matches with per-frame
BallStatus / BallPossession flags from DFL's tracking feed. Bassek, M., Rein, R., Weber, H., Memmert, D.
(2025), "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12;
figshare article 28196177 v1, https://doi.org/10.6084/m9.figshare.28196177. Data (c) DFL / IDSSE, CC BY 4.0.
None of it is the competition match or Metrica. `lde_frozen/PROVENANCE.md` has the file md5s, and
`lde_frozen/rebuild/rebuild_models.sh` is the vendored one-command rebuild (fetch -> make_dataset
`--drop-p 0 --blackout-p 0` -> train `d,ha,c` -> `cmp` against the frozen files).

## Results

Dev = Sample Games 1+2 x 2 seeds (shared bench). "off" = AR(1) ball offset (sd 1.3 m, tau 1 s); "s2" =
out-of-distribution stress (sd 2 m, tau 3 s, plus extra 0.5-2 s ball dropouts); "osd" = offset + smoothing
+ player dropout (all from `tuning/sweep_r2.py`). "DFL" = `bench.py --dfl` leave-one-game-out on the 7 DFL
games with tracking-flag labels (the real ground truth's meaning), using the judge's round-1 fold map
`candidates/_scratch/judge_measured-performance/r3/folds_r1.json`.

| version | spellF1 | macroF1 | off | s2 | osd | DFL | match possessions / dead share |
|---|---|---|---|---|---|---|---|
| round 0 (`tuning/archive/pipeline_r0.py`) | 0.6035 | 0.8121 | 0.5202 | | | | 269 / 0.449 |
| round 1 as reported (old LDE models, no longer reproducible) | 0.6470 | 0.8119 | 0.5855 | | | | 316 / 0.460 |
| round 1 (`tuning/archive/pipeline_r1.py`) on the frozen models | 0.6429 | 0.8106 | 0.5786 | 0.5398 | 0.5677 | | 300 / 0.459 |
| round 2 (`pipeline_r2.py`) | 0.6479 | 0.8111 | 0.5957 | 0.5466 | 0.5774 | 0.5423 | 317 / 0.458 |
| **round 3 (`pipeline.py`)** | **0.6632** | **0.8109** | **0.6085** | **0.5432** | **0.5809** | **0.5987** | **335 / 0.469** |

Round 3 bench rows: 0.6561 / 0.6832 / 0.6606 / 0.6528. DFL per game: 0.5114 / 0.5393 / 0.6386 / 0.6074 /
0.6179 / 0.6302 / 0.6459 (macroF1 0.9028). `submission.csv` md5 99b8a7b62623c4c18ae4af6d3562237b, the same
from `bench.py --match` and from `make_submission.py` (with a different PYTHONHASHSEED); `evaluation/validate_submission.py`: Valid.

## Round 3 changes (from the judges' feedback)

Kept:
- **Stage-A dead logit replaced with v2's learned form** (all five judges): `a_hand` 1, `w_lde_dead` 1,
  `a_bias` / `a_slow` / `a_few` / `a_static` 0, keeping `a_out` 4, `a_long` 4 and the tag term. Dev
  0.6479 -> 0.6555 (+0.008), DFL 0.5423 -> 0.5813 (+0.039, 7/7 games up). The hand-heavy S2 logit was fitted to
  event-derived Metrica labels, whose dead-ball meaning differs from the tracking-flag GT. It also drops
  four Metrica-calibrated constants.
- **Local contrast of the learned team logit** (measured-performance judge): `lc_beta` 0.75, `lc_gate` 1.5,
  `lc_hw` 100 frames, applied in `decode()`. Dev 0.6555 -> 0.6647 (+0.009), off +0.0065, s2 -0.0004, osd +0.0044,
  DFL 0.5813 -> 0.5938 (7/7 games up). Grid beta {0.5, 0.75, 1, 1.5} x gate {0.5, 1, 1.5} x hw {50, 100, 200}:
  gate 0.5 (the judge's suggestion) reaches 0.678 on clean dev, but it loses 0.016 on osd and 0.009 on s2,
  the match-like views. 0.75 / 1.5 / 100 is the only setting non-negative on every view, so I took it
  over the higher dev number.
- **Soft long-gap gate** `lm_ramp` 0.2, `lm_gate` 1.75 (generalisation judge, robustness): dev -0.0015 (noise),
  s2 +0.003, osd +0.001, DFL +0.0004 (with the k cap). Label-free on the match, moving the gate down by 0.25 m/s now changes the
  dead share by 0.9 points instead of 1.9 with the hard gate (frame agreement 0.981 -> 0.988). In round 2, with the
  old hand-heavy blend, the ramp had made it *more* sensitive; with the learned dead logit it no longer does.
- **Snap radius k capped at 1** (`snap_k_cap` 1.0; generalisation and method judges): no change on dev by
  construction (k = 1 there), DFL +0.0045 (5/7 games up, 1 equal, 1 down by 0.002). It removes the last match-adaptive
  extrapolation (match k 1.72) from the decode.
- **Reproducibility (rules judge):** `make_submission.py`, a standalone one-command writer that does not
  need the Metrica sample data and refuses to run with `HTD_SWEEP=1`;
  `lde_frozen/MANIFEST.json` (the loader reads md5s and the lightgbm version from it) and `PROVENANCE.md`
  with the dataset citation; vendored rebuild scripts in `lde_frozen/rebuild/`, verified in round 3 to
  reproduce `model_d`, `model_ha` and `model_c` byte for byte from the cached IDSSE npz; `pipeline_r0.py` / `pipeline_r1.py` moved to `tuning/archive/`
  with a "not the submission" header; `.python-version` 3.14.6.

Rejected (re-measured on all views):
- Learned hand-over switch cut `a_c` 0.5 / 0.3, `g_c` 3 on top of the new stage A: every view went down
  (clean -0.001, off -0.002 to -0.003, osd -0.003 to -0.004).
- `a_vel` 0 (drop the velocity-reversal cut): bench-neutral, DFL -0.0026. Kept as it was.
- `dmin_D_s` 1: noise (-0.0002 clean, -0.002 s2).
- Not attempted: the explicit-duration (HSMM) stage B with a DFL duration prior, and a DFL-fitted dead
  stacker. Stage-B decoder knobs stay frozen, since the judges measured them flat (+-0.003).

## Round 2 changes (from the judges' feedback)

Kept:
- **Frozen LDE dependency** (rules/reproducibility fix). `_lde_module()`, which imported the sibling's live
  `pipeline.py` and models, is replaced by a local loader on `lde_frozen/` with md5 and version checks. It
  reproduces the current code exactly (0.6429). The reported round-1 0.647 / 316 possessions came from
  models that were retrained after round 1 and no longer exist.
- **Hand-built proximity terms off** (`w_shape`, `w_near`, `w_carry` 0) and `dmin_HA` 12 -> 25:
  +0.004 clean, +0.013 off, +0.007 s2, +0.006 osd. These terms are strongly correlated with the learned
  logit (double counting) and were the part that broke under the offset.
- **`prune_uncontrolled` off**: +0.001 clean, +0.003 off, +0.001 s2, +0.001 osd. It removes a fixed 1.5k
  proximity rule. Kept as a robustness and simplicity change.
- **Touch-onset gating off** (`a_touch` 0): bench-neutral (-0.000 clean, +0.001 off, -0.001 s2, +0.003 osd),
  but it removes the last k-scaled radii from the decoder. Label-free check on the match: the spellF1
  agreement between the output at the estimated k (1.72) and at k = 1 goes 0.930 (round 1) -> 0.975
  (proximity terms off) -> 0.989 (touch gating off as well).
- `requirements.txt` with pinned versions; `tuning/_cache/` and `tuning/_cache_r2/` are git-ignored.

Rejected (within noise or worse, measured on all four views):
- `w_lde` 3/5/6; `base_sw` 3/5/6; `a_vel` 0/0.5; `dmin_HA` 8/18/37; `min_spell_HA_s` 0 (D-only merge);
  `dmin_D_s` 3; `snap_win_s` 0 (-0.008) and 1.0; `snap_r` 1.0/2.5.
- Learned hand-over detector in the switch cost (`a_c` 0.3/0.5/0.8, `g_c` 3/6): best +0.003 clean and
  +0.004 off, but -0.002 to -0.004 on s2. Left in the code, off.
- Stage A: `a_hand` 0.7 + `w_lde_dead` 0.5 (-0.002 clean, -0.003 macroF1); `a_long` 2 (-0.012); a soft
  long-gap gate (`lm_ramp` 0.25/0.5: -0.001/-0.005 clean, and it made the match dead share *more*
  sensitive to `lm_gate`: 0.442-0.456 vs a constant 0.458 for `lm_gate` 1.75-2.25 with the hard gate);
  non-additive few-tracked/long-gap cue (`few_long` "max": bit-identical on the bench, no change on the
  match). All left in the code, off.
- Not attempted: the DFL-fitted stacker / stage-A fit, the joint 3-label decode and the explicit-duration
  stage B. The judges' oracles put the remaining under-splitting in the evidence, not the decoder.

## Round 1 changes (from the judges' feedback)

Kept because they improved dev spellF1 by more than 0.003:
- Learned team log-odds from `learned_dfl_evidence` added to stage B (`w_lde` 4): +0.025 clean, +0.056 offset.
- Stage-A dead logit blended with the learned p_dead (`a_hand` 0.85, `w_lde_dead` 0.3): +0.004.
- `dmin_HA` 37 -> 12 frames: +0.005. Tracking-flag ground truth has many spells under 1 s.
- Team snap window `snap_win_s` 1.5 -> 0.5 s: +0.009 clean, +0.016 offset. The learned evidence already
  places the change, so the wide first-touch snap was moving it too late.

Kept as rules and robustness fixes (match only, so no bench effect):
- The Smart Tagging team prior uses only the phase codes, as the sign of their net vote, at weight 0.2.
  Before, it used v1's 12 overlapping codes at weight 0.8, which could not be validated and removed
  possessions on the match. The set-piece team prior is dropped.
- The kick-off pre-D search window is 180 s instead of 10 min.
- k is computed over graded frames only, and `k_hi` is 2.5 instead of 2.0.
- The team-name substrings (`barc` / `atl` / `athl`) are removed from `side_of`.
- The `HTD_CFG` env override is read only when `HTD_SWEEP=1`, and the overrides it applies are printed
  to stderr.
- The merge loop is bounded by the path length (each pass removes a segment).

Rejected:
- Sandwich states with learned entry evidence: +0.013 clean but -0.013 under the offset.
- `base_sw` 3/5, `a_vel` 0.5 and `min_spell_s` 0.3: within noise.
- The re-check at the final point was flat: `w_lde`, `w_shape`, `dmin_HA`, `w_carry`, `w_near`,
  `a_touch` and the dead blend all landed within 0.004.

## Dependencies and reproduction

- `requirements.txt`: numpy 2.5.4, pandas 3.1.1, lightgbm 4.7.0; Python 3.14.6 (`.python-version`).
- **Submission:** `python -I candidates/hier_touch_decoder/make_submission.py` reads the match through
  `games.require` and writes `submission.csv` (md5 99b8a7b62623c4c18ae4af6d3562237b) in about 15 s. It does not
  need the dev bench data. `python candidates/bench.py candidates/hier_touch_decoder/pipeline.py --match`
  writes the same file after the dev bench.
- The output is deterministic: no RNG, LightGBM single-threaded, and byte-identical across runs and
  PYTHONHASHSEED values.
- Models: `sh candidates/hier_touch_decoder/lde_frozen/rebuild/rebuild_models.sh` (see `lde_frozen/PROVENANCE.md`).
- Tuning sweep: `python candidates/hier_touch_decoder/tuning/sweep_r2.py '[{...cfg overrides...}]'` scores
  the dev rows on the four views (clean, off, s2, osd) and caches the features and the raw learned
  probabilities to `tuning/_cache_r2` (git-ignored; the contrast is applied in `decode`, so the cache stays
  valid). DFL check:
  `HTD_SWEEP=1 HTD_CFG='{...}' python candidates/bench.py candidates/hier_touch_decoder/pipeline.py --dfl --folds candidates/_scratch/judge_measured-performance/r3/folds_r1.json`.
  `pipeline_r2.py` is the round-2 backup, and `tuning/archive/` holds the older revisions (not the submission).
