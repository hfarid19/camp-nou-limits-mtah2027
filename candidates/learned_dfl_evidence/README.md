# learned_dfl_evidence

LightGBM frame evidence (dead/alive, home/away, hand-over detector), trained only on the open
DFL / IDSSE tracking data (7 Bundesliga matches, per-frame BallStatus / BallPossession flags,
Bassek et al., Sci Data 2025, figshare 28196177 version 1, CC BY 4.0, credit DFL Deutsche
Fussball Liga / IDSSE). The data is degraded with domain randomisation so it looks like the
competition tracking, and the evidence feeds possession_pipeline's 3-state Viterbi. The full
write-up is in the pipeline.py docstring.

## Training data provenance (RULES.md 1: human-touched data from other games)

- What: per-frame `BallStatus` (alive / dead) and `BallPossession` (home / away) flags from DFL's
  official tracking feed (`positions_raw_observed` XML), 25 fps.
- Who and how: produced by DFL's tracking provider, semi-automatically with operator quality
  assurance. They are not hand-labelled by us, and we did not edit them.
- How much: 7 matches (Bundesliga / 2. Bundesliga 2022/23), 1,002,644 graded frames (11.1 h),
  3,061 possession spells (412 per 90 min), dead share 0.448. Training uses a stride-5
  subsample of these frames, degraded with 3 seeds.
- Where from: the open dataset of Bassek et al., Sci Data 2025 (figshare 28196177 v1, CC BY 4.0).
  `fetch_idsse.py` downloads it and checks every file's md5.
- Nothing is labelled on the competition match. The labels mean the same thing as the
  competition ground truth: the team the tracking names as owning the ball, and D where the
  tracking marks the ball dead.

## Rebuild

The raw data and caches go to `$IDSSE_CACHE`, or `./external/idsse` (git-ignored) by default.
`--cache` overrides it. The figshare article version is pinned, and every downloaded file is
checked against the committed `idsse_manifest.json` (id, name, size, md5).

One command does the whole rebuild and ends with `cmp` against the committed model files:

    pip install -r candidates/learned_dfl_evidence/requirements.txt
    sh candidates/learned_dfl_evidence/rebuild_models.sh [OUT_DIR]

The download is 21 files, about 2.6 GB, and needs network access. Rough times on one laptop
(18 cores): fetch and parse 10-20 min (network bound), each `make_dataset` about 1 min, train
ha about 4 min, train d,c about 3 min.

The steps it runs:

    python -I fetch_idsse.py --games 7 --cache C                     # download + md5 + parse -> C/npz
    python -I make_dataset.py --cache C/r1 --drop-p 0 --blackout-p 0  # round-1 dataset (team model)
    python -I make_dataset.py --cache C/r2                            # round-2 dataset (dead, hand-over)
    python -I train.py --cache C/r1 --out OUT --models ha
    python -I train.py --cache C/r2 --out OUT --models d,c
    python -I dfl_logo_eval.py --fold-dirs '{"ha": "C/r1/folds"}' --cache C/r2   # diagnostic only

Each `ds/config.json` records its degradation settings, and they are copied into
`model_meta.json["datasets"]`. `train.py` writes a sha256 for each model file into
`model_meta.json["sha256"]`, and `pipeline._models()` asserts those hashes at load. A retrain
that does not also update the meta fails loudly instead of silently changing a submission.

To write the submission without the dev bench:
`python -I candidates/learned_dfl_evidence/make_submission.py`. Its output is byte-identical to
`bench.py --match`.

Dependencies (requirements.txt): lightgbm==4.7.0, numpy==2.5.4, pandas==3.1.1 (Python 3.14.6).
The pipeline asserts the installed lightgbm version against `model_meta.json["lgb_version"]`.
It is deterministic: num_threads=1, fixed seeds and crc32 game keys. train.py falls back to a
serial loop where `fork` is unavailable (Windows).

Decoder constants live in `CFG`. The `LDE_<KEY>` environment overrides are honoured only when
`LDE_SWEEP=1` (tuning only), and then a warning naming every overridden key goes to stderr. An
`LDE_*` variable without `LDE_SWEEP=1` is ignored, also with a stderr warning, and
`make_submission.py` refuses to run while any `LDE_*` variable is set. Tuning scripts in `tuning/`:
- `r3_views.py`: the round-3 harness. It caches frame probabilities once on the dev rows under
  four views (clean, offset, osd, the judges' stress2) and on the DFL leave-one-game-out rows
  (fold models from the `bench --folds` JSON in `$LDE_FOLDS`), then scores `decode()` configs on
  all five.
- `r2_cache.py`: round 2. `sweep.py`: round 0.
- `archive/`: round 0's `ablate.py` / `diag.py` and round 1's `r1_*.py`. They are superseded and
  not part of the documented flow.

Caches go to `$LDE_TUNE_CACHE` or `./external/tune`. `.python-version` pins 3.14.6.

## Round 3 changes (judges' feedback)

Same keep rule: dev spellF1 up by more than 0.003 with macroF1 not falling more than 0.01, or a
fix for a rules, determinism or robustness problem a judge raised. Every config was scored on
five views with `tuning/r3_views.py`: the dev rows clean, `offset`, `osd` (offset + smooth +
player dropout) and the judges' `stress2`, plus DFL leave-one-game-out. DFL uses `bench --dfl`
rows: degradation seed 2027, the fold models that never saw the scored game, and tracking-flag
labels with the same meaning as the GT. The "clean" column is the official dev bench, and each
kept step was re-run with `candidates/bench.py`.

| step | clean / macroF1 | offset | osd | stress2 | DFL LOGO | decision |
|---|---|---|---|---|---|---|
| round 2 | 0.6502 / 0.8118 | 0.6095 | 0.5888 | 0.5387 | 0.5929 | |
| `merge_short_v2`: shortest offending segment first, segments recomputed after every merge (replaces `pp.merge_short`'s order-dependent left-to-right 5-pass walk) | 0.6521 / 0.8116 | 0.6130 | 0.5916 | 0.5414 | 0.5933 | kept (soundness fix a judge raised; up on every view) |
| v2's D post chain instead of `snap_dead_edges`: D end to the nearest long (> 3 s) ball-gap end within `snap_win_s`, in either direction; alive-to-D start moved back to the first ball-out frame within `out_lookback_s` (staying `keep_s` 0.5 s inside the possession). Windows 1 s / 6 s, see below | 0.6547 / 0.8135 | 0.6160 | 0.5954 | 0.5469 | 0.6036 | kept (robustness: fixes the DFL D-end tail a judge raised; +0.0045 over round 2, up on every view) |
| local-contrast team logit: `l' = l + 1.25 * sign(c) * max(abs(c) - 1.25, 0)`, with `c = l - moving_mean(l, 100 frames)` and `l = logit(p_home)` | **0.6725 / 0.8145** | 0.6204 | 0.5882 | 0.5436 | 0.6243 | kept (+0.018 dev, +0.021 DFL) |
| `w_tag_team` 0.4 -> 0.1 (Smart Tagging build up / progression team prior) | no-op | | | | | kept (robustness: the bench can't test it, and its only effect was to merge short opposing touches inside phase tags of about 18 s; match 315 -> 325 possessions) |
| fail-closed loading: `model_c` mandatory; `model_meta.json` must contain `lgb_version` and a sha256 for each of d, ha, c; LDE_* warnings; `make_submission.py` refuses LDE_* variables; `.python-version`; tuning clutter moved to `tuning/archive/` | no-op | | | | | kept (rules) |

**D-end tail (DFL).** v2's chain with its own windows (4 s / 12 s) left DFL flat at 0.5929. The
reason is that a 4 s window pulls a D end to a gap end that is not the restart. DFL then rises
monotonically as the window shrinks: 4 s 0.5929, 3 s 0.5955, 2 s 0.6003, 1 s 0.6036, 0.5 s
0.6040. At 0 s the dev views drop (clean 0.6538). Dev is flat from 1 s to 4 s, so 1 s is the
plateau edge that is safe on both. Look-back 6 s against 12 s: offset +0.003, other views flat.

Effect on the DFL D ends, measured as the predicted D end minus the nearest GT D end within 5 s:

| | mean | p90 | share more than 1 s off |
|---|---|---|---|
| round 2 | +0.46 s | +1.60 s | 0.198 |
| round 3 | +0.11 s | +0.68 s | 0.095 |

**Contrast plateau.** The table shows beta / gate with the post chain on.

| beta / gate | clean | offset | osd | stress2 | DFL |
|---|---|---|---|---|---|
| 1.0 / 1.0 | 0.6745 | 0.6190 | 0.5856 | 0.5407 | 0.6267 |
| 1.25 / 1.25 (kept) | 0.6725 | 0.6204 | 0.5882 | 0.5436 | 0.6243 |
| 1.5 / 1.5 | 0.6680 | 0.6216 | 0.5901 | 0.5467 | 0.6202 |

The gate matters. Without it the term over-reacts on the noisy views: beta 0.75 with gate 0.5
gives osd -0.010. A half width of 50 frames loses most of the gain, and 150 frames matches 100.
Pre-smoothing the logit before the contrast (half width 3 or 6) is slightly worse.

On the match, possessions go from 308 to 315 with the post chain and contrast, and to 325 with
`w_tag_team` 0.1. The dead share moves from 0.474 to 0.472. DFL labels have 412 spells per
90 min, so the pipeline still under-splits: 2202 predicted against 3061 on DFL LOGO.

Rejected (measured, reverted):
- `w_out` 4 (v2's ball-out dead term): clean +0.0006 and stress2 -0.002, so noise.
- `refine_window_s` 0.25 / 0.35 / 0.75 (a judge asked to extend the sweep below the edge). At
  0.35: clean +0.0017, osd +0.0016, DFL +0.0023, all within noise and under the keep bar. At
  0.75: clean -0.003. 0.5 sits on a plateau, so it stays.
- `switch_cost` 0.5 / 1.0, `min_poss_s` 0.6 and `gamma` 1, all on the new chain: clean
  -0.003 to +0.001. `min_poss_s` 0.6 also cost osd -0.014.
- `w_gap` 1.5 / 0 (a judge suggested a stacker). On out-of-fold DFL with degradation seed 21,
  the fitted weight of the long-gap term next to `logit p_dead` is about -0.8, and on DFL LOGO
  w_gap 0 gives +0.0035. But dev drops on every view: clean -0.001 / -0.004, s2 -0.002 / -0.007.
  The bench degradation lacks the match's long blackouts, so the term stays at 3.
- Separate `tau_d` / `tau_h` (Platt scaling). Out-of-fold DFL Platt slopes are 0.95 for the
  dead logit and 0.97 for the team logit, so the two models are equally calibrated and a split
  would change nothing.
- Short-touch temporal stacker (the judges' "biggest untried idea"). This is a LightGBM on
  out-of-fold DFL `p_home` at offsets up to +-3 s, its contrasts, the `p_change` window and the
  dead logit, with team-swap augmentation and symmetric inference. Trained on the bench's own DFL
  degradation family (seed 21) it looked good: clean 0.6801, DFL 0.6316. Built the shippable way
  (the candidate's `make_dataset.degrade_dfl`, seed 31, per-fold stackers for an honest LOGO), it
  gave clean 0.6741 (+0.0016), osd +0.008 and s2 +0.009, but DFL 0.6100 (-0.014). The first
  result was in-distribution fitting to the bench degradation, so the stacker was not shipped.
  Script: `candidates/_scratch/lde_r3/build_stack_rejected.py`.

## Round 2 changes (judges' feedback)

Same keep rule as round 1. Each change was kept only if dev spellF1 rose by more than 0.003
with macroF1 not falling more than 0.01, or if it fixed a rules, determinism or robustness
problem a judge raised. Every step was also scored on four views of the dev rows: clean, `osd`,
`offset`, and the judges' out-of-distribution `s2` (AR(1) offset sd 2 m, tau 3 s, plus extra
0.5-2 s dropouts). The cached-probability harness is `tuning/r2_cache.py`, which adds `s2`, and
`r1_ev.py`. The DFL leave-one-game-out co-check now runs on two degradations: `r1d` (the round-1
setting, for comparison with 0.5409) and `bo` (stoppage blackouts on every game, which gives
match-like long gaps).

| step | clean / macroF1 | osd | offset | s2 | DFL r1d / bo | decision |
|---|---|---|---|---|---|---|
| round 1 | 0.6467 / 0.8098 | 0.5786 | 0.6052 | 0.5273 | 0.5409 / 0.5498 | |
| `refine_window_s` 1.0 -> 0.5 | 0.6508 / 0.8105 | 0.5792 | 0.6028 | 0.5334 | 0.5459 / - | kept (+0.004; DFL agrees) |
| soft long-gap ramp: `w_gap * sigmoid((1.75 - mean speed)/0.2) * clip((len - 2 s)/3 s, 0, 1)` instead of the hard `len > 3 s and speed < 2.0` gate | 0.6505 / 0.8097 | 0.5798 | 0.6038 | 0.5389 | | kept (robustness: removes a knife edge tuned only on Metrica; match dead share 0.477 -> 0.471) |
| retrained `model_d` + `model_c` on the round-2 dataset: extra 0.5-2 s ball dropouts and stoppage blackouts on half of the degradations each (`model_ha` unchanged) | 0.6516 / 0.8122 | 0.5847 | 0.6109 | 0.5456 | 0.5446 / 0.5567 | kept (robustness: up on every stress view, clean flat) |
| `w_gap` 4 -> 3 (the dead model now learns long gaps itself: w_gap 0 now costs 0.009, against 0.02+ before) | **0.6502 / 0.8118** | 0.5837 | 0.6095 | 0.5445 | 0.5446 / 0.5581 | kept (calibration, which a judge raised; flat on every view) |
| rules and reproducibility: model sha256 in meta, asserted at load; `train.py --models`; `ds/config.json` provenance; `make_submission.py`; `rebuild_models.sh`; `candidates/_scratch/` in the root .gitignore; `tuning/_cache`, `external/tune`, `*.pkl` in the candidate .gitignore; provenance paragraph | no-op | | | | | kept |

Net change from round 1: clean +0.0035, osd +0.005, offset +0.004, s2 +0.017, DFL LOGO +0.004
(r1d) and +0.008 (bo).

The stoppage blackout (`make_dataset._blackout`, own RNG stream):
- On blackout degradations, the bench-style random gaps are scaled down to 25-60% of their
  target.
- The ball is hidden over whole BallStatus-0 stoppages with probability U(0.5, 0.95). The edges
  are jittered by -0.5..+1 s at the start and -1..+0.5 s at the end.
- Live windows (exponential, mean 4 s, at least 1 s) are hidden on U(2, 8)% of alive frames.
- Inside 30% of the hidden stoppages, the tracked players collapse to 5-15.

This brings the long-gap structure that the match input shows (label-free: 41% of graded
frames inside gaps > 3 s) into training, so the dead model learns P(D | gap) from tracking-flag
labels instead of relying only on the Metrica-tuned `w_gap`.

Rejected (measured, reverted):
- Wider ball offset in training for all three models (sd U(0, 2) m, tau U(1, 3) s, with or
  without the extra dropouts): s2 +0.012 to +0.020, but clean -0.016 to -0.020. Swapping models
  one at a time showed the whole clean loss is the team model (home from the wide-offset dataset:
  clean -0.012 to -0.017). The wide-offset dead + hand-over models with the round-1 team model
  ('B': clean 0.6486, osd 0.5911, offset 0.6106, s2 0.5498, DFL r1d 0.5424) were close to the
  kept set. The kept set ('D', round-1 offset + dropouts + blackouts) won on clean and on DFL r1d.
- `w_gap` 2 (method-soundness suggestion): clean -0.004, osd -0.003 with the new models.
  `w_gap` 0: -0.009.
- `gamma` 1 (hand-over detector as a switch-cost likelihood ratio): within +-0.002 on all views.
- `switch_cost` 0.5 / 1.0: clean -0.001 / -0.005, mixed stress.
- `d_end_snap_s` 0 or 2 (cap on late-restart D extensions): within +-0.002.
- `min_poss_s` 0.5 / 0.75: within +-0.001.
- `gap_speed_ms` 1.75 with the hard gate, or ramp widths 0.1: same as the kept ramp, within noise.

Not attempted this round: a stacked hand-over / short-touch head (`model_c` on +-1 s windows of
out-of-fold p_home) and an exact minimum-duration decoder. Both target the under-split (about
266 predicted spells against 338 GT on the bench), which remains the largest gap.

Siblings: `hier_touch_decoder` and `v2_stacked_fixes` use md5-pinned frozen copies of the round-1
models (`lde_frozen/`), so this retrain does not change their outputs.

## Round 1 changes (judges' feedback)

Each step was benched, and kept only if dev spellF1 rose by more than 0.003 with macroF1 not
falling more than 0.01, or if it fixed a rules or robustness issue a judge raised. Robustness
was also checked on two stressed copies of the bench (fixed seeds):
- `osd`: the match-like offset + smooth + player dropout.
- `offset`: the AR(1) ball offset (sd 1.3 m).

| step | dev spellF1 / macroF1 | osd | offset | decision |
|---|---|---|---|---|
| round 0 | 0.6129 / 0.7988 | 0.539 | 0.569 | |
| + gated long-gap dead term (`w_gap` 4: raw-ball gap > 3 s, mean player speed < 2 m/s) | 0.6332 / 0.8088 | 0.561 | 0.590 | kept |
| + `refine_window_s` 1.5 -> 1.0 (hand-over peak snap window) | 0.6400 / 0.8099 | 0.574 | 0.600 | kept |
| + Smart Tagging fixes: D before each half's `kick off (start)` tag; goal dead prior on the long ball gap nearest the goal tag's end (was a fixed 10 s window after it, i.e. live play after the kick-off); exact team-name matching (no 'barc'/'atl' fallbacks); hidden `2.0` factor folded into `w_tag_team` 0.4 (output identical); `LDE_SWEEP` gate | 0.6400 / 0.8099 (bench no-op) | | | kept (rules / match) |
| + training augmentation: low-pass the degraded ball (9-13 frame NaN-aware mean) on half of the degradations, separate RNG stream; retrained all 3 models | **0.6467 / 0.8098** | 0.579 | 0.605 | kept |
| reproducibility: cache path from `$IDSSE_CACHE` / `./external`, pinned figshare version + md5 manifest, pinned requirements + lightgbm version assert, `fork` fallback, tuning scripts in repo | no-op | | | kept |

On the match, the tag fixes change 1,517 frames. About 114 s before the 2nd-half kick-off
becomes D (about 8 spurious spells removed). At the after-goal kick-offs, the D that was being
pushed onto live play is gone. The 2nd-half kick-off tag has no clean restart inside it (the ball
is tracked moving throughout), so D stops at the tag start rather than 6.6 s later.

Rejected (measured, reverted):
- Blending HTD's hand-built dead logit into p_dead (a 0.3-1.0 blend, or the hard stage-A mask):
  0.630-0.636 clean. w_gap already captures it.
- Joint re-grid of switch 0.5-1.25 x dead 4-6, on clean and on osd: everything within +-0.002.
- `gamma` 1-2 (the hand-over detector on the switch cost): +0.006 clean, -0.003 osd.
- `refine_window_s` 0.75 after the retrain: +0.004 clean, flat on both stress views. Treated as
  noise on a plateau.
- Dead-prior removal (scaled likelihoods), a +-0.2..1.0 bias on the dead log-odds: flat.
- High-pass of the learned team logit, to split H-A(short)-H sandwiches: +0.017 clean but
  -0.03 to -0.06 under stress. This is the same trap as the proximity splits.

Diagnostic (not a selection target): inside short (1-3 s) sandwiched GT opponent spells, the
team logit rises from about -1.4 to about 0 but does not cross. About half of these spells are
still swallowed. That is the next evidence target, for example a dedicated short-touch head.

## Results

DFL leave-one-game-out diagnostic (round-1 fold models, unseen degradation seed 21,
tracking-flag labels, not a selection target): spellF1 0.5409, macroF1 0.8938 (round 0: 0.527 /
0.893). Per game: 0.54 / 0.50 / 0.65 / 0.58 / 0.58 / 0.61 / 0.33 (J03WR9 remains the outlier).

Final bench, round 3 (`candidates/bench.py ... --match`):

    RESULT spellF1_mean 0.6725, macroF1_mean 0.8145
      Sample_Game_1 2027 0.6918 / 0.8366  288/348     Sample_Game_1 2028 0.6934 / 0.8353  301/348
      Sample_Game_2 2027 0.6549 / 0.7934  272/329     Sample_Game_2 2028 0.6501 / 0.7928  266/329
    match: 325 possessions, mean 10.3 s, dead share 0.472, no X inside the halves
    submission.csv md5 c41c7139090fde7a0dcf326de48e0295 (bench --match and make_submission.py
    with PYTHONHASHSEED=12345 are byte-identical); validate_submission.py: OK

`bench.py --dfl` with fold models (folds: d, c -> round-2 folds, ha -> round-1 folds; seed 2027):
spellF1 0.6243, macroF1 0.9144 (round 2: 0.5929). Per game 0.558 / 0.578 / 0.662 / 0.620 /
0.641 / 0.642 / 0.671.

Final bench, round 2 (for reference): 0.6502 / 0.8118, match 308 possessions, dead share 0.474,
md5 e4f53ef786f5b15c53298e09b3f8b920.

DFL leave-one-game-out with the final models' fold models (seed 21): 0.5446 / macroF1 0.8947
on the round-1 degradation, and 0.5581 / 0.8995 with stoppage blackouts.

Round 1 (for reference): 0.6467 / 0.8098, with 305 possessions and dead share 0.477 on the
match. Round 0 was 0.6129 / 0.7988, with 317 possessions and 43.7% dead on the match. The match
dead share rose with `w_gap`, which marks long untracked slow stoppages as D, and then with the
pre-kick-off D. The soft ramp and `w_gap` 3 bring it back to 0.474. That is still about 2.6
points above DFL's tracking-flag share of 44.8%.

Reproducibility checks (round 1; the round-2 rebuild check is below):
- Two `--match` runs (PYTHONHASHSEED unset and 12345) are byte-identical
  (md5 91977a67babc9d3353966a9b9da6c8f2), and `evaluation/validate_submission.py` reports OK.
- `make_dataset.py` rebuilt from the npz gives bit-equal X/y/yc/idx for all 21 game x seed files.
- `train.py` retrained on that rebuild gives `model_{d,ha,c}.txt` and `model_meta.json`
  byte-identical (cmp) to the committed ones.


Round-2 rebuild check: both datasets rebuilt from the npz are bit-equal to the ones used for
training (X/y/yc/idx, 21 + 21 files). Run on that rebuild, `train.py --models ha` and then
`--models d,c` give `model_{d,ha,c}.txt` byte-identical (cmp) to the committed files. The
committed `model_meta.json` is the one that sequence writes. `evaluation/validate_submission.py`
reports OK on the round-2 submission.csv.

Round-3 rebuild check: `rebuild_models.sh` was run end to end into a fresh cache, with the npz
already parsed, so no download was needed. Both `make_dataset` runs, `train.py --models ha`, then
`--models d,c` took 8 min 06 s wall on 18 cores. All four of `model_d.txt`, `model_ha.txt`,
`model_c.txt` and `model_meta.json` were reported "identical to the committed file" by `cmp`.
That includes the round-2 `model_d` and `model_c` the submission depends on. Shipped sha256
prefixes: d f62083d8, ha 7ad6ba2e, c c5011bcf.
