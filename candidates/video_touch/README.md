# video_touch: a video touch detector for possession boundaries (built, measured, NOT recommended)

**Question.** The tracking can't tell an interception from a deflection. Can a small vision model,
run locally on the tactical video, detect touches (and their team) well enough to fix short spells
and boundaries? **Answer on held-out SoccerTrack v2 games: no.** The detector finds touches well
above chance, but its touch evidence doesn't separate right from wrong short spells, and every
possession rule built on it scores at or below `ensemble_ctrl_v5c`.

## Data (SoccerTrack v2, CC BY 4.0, release v1.2)

10 university matches, fixed 4K panorama (4096x1080; 132831 is 3840 wide), 25 fps, 21,428 ball-action
events. Touch = every event except OUT and GOAL (19.1k). Alignment: video frame count equals the
ball rows (67,625 for 117093 1st). Event `frame` is 1-based (video frame = `frame - 1`). On held-out
games the detector's matched offsets have median 0 (mean +0.3 frames).
Pitch->pixel map: a degree-4 polynomial fitted per half on GSR player pairs, from 3 MB range-fetched
slices of each GSR JSON (`fetch_gsr_slices.py`, `st_geom.py`). Median error 9 px horizontally, 1.5 px vertically.

**Crops** (`extract_st.py`, `extract_match.py`): 14 m (horizontal) square around the ball,
resized to 160x160 RGB, every frame. The centre follows the ball track smoothed with sigma = 1 s, because
SoccerTrack's ball is interpolated between the annotated events; an unsmoothed crop path would
leak the event timing. The match uses the tracking ball through the per-frame homography. On frames v5c
calls alive, the match ball is present 96% of the time.

## Model and training

`vt_model.py`: timm ResNet-18 (ImageNet) with temporal shift in every block, plus a dilated
temporal-conv head that gives per-frame touch logits. Input is 32-frame clips (64 at inference), 128 px crops.
BCE loss with soft labels (1 at the event, 0.5 at ±1 frame) and pos_weight 5. Training: 6,000 steps, batch 12,
AdamW with one-cycle, bf16 on MPS (about 50 min per fold on a shared GPU). Evaluation is 2-fold and game-grouped:
A tests on 117092/118575/118577/128057/132831, B on the other five.

## Results on 10 held-out games (peaks: NMS radius 6; `eval_touch.py`)

| | AP | P / R at thr 0.4 | P / R at thr 0.5 | chance AP |
|---|---|---|---|---|
| ±5 frames | 0.443 | 0.475 / 0.431 | 0.589 / 0.272 | 0.148 |
| ±2 frames | 0.169 | 0.273 / 0.248 | 0.351 / 0.162 | 0.047 |

Interception-type events are the weakest (prototype half: block 25%, tackle 10% recall).
Team attribution of detected touches (nearest tracked player to the ball): 72-78% correct per game.

## Phase 3: possession rules vs ensemble_ctrl_v5c (`st_phase3.py`, nested leave-one-game-out on fine3)

v5c codes come from `candidates/soccertrack/bench_st.py` (degrade_v2, seed 2027); labels are `bas_all`.

| rule | simulated perfect touches, tracking team | real detector |
|---|---|---|
| snap H<->A switches to touches of the new team | +0.0042 | -0.0001 |
| insert opponent spell at opponent touches | -0.039 | -0.0053 |
| remove short spell with no touch by its team | +0.0100 (+0.031 with true team) | **-0.0025 (2/10 games)** |
| control: remove all such spells, no video | +0.0074 | +0.0074 |

The video gate makes "remove" worse than removing everything. Spell-level AUC of "video sees a
touch in the spell" for spell correctness is 0.485 (duration alone: 0.561).

## Cost and determinism (measured under heavy machine load)

- Match crops: 199,319 frames decoded and cropped in 1,527 s (CPU, PyAV 19.0.2). Re-extraction is byte-identical.
- CNN on CPU: 26-58 fps, so about 1-2 h per model for the full match, about 100 s for v5c's 20 candidate
  spells. Outputs are identical across repeated runs and 3/4/8 threads (`torch.use_deterministic_algorithms`).
  MPS: about 2,800 fps, but it differs from CPU by about 3e-8, so it isn't byte-safe.
- Shipping would need torch + timm + PyAV + opencv and the 45 MB weights. Not recommended given the result above.

## Reproduce

All scripts are in `$MTAH_WORK/video_touch/scripts` (copies here). In order: `fetch_gsr_slices.py`, `st_geom.py`,
`extract_st.py <m>_<h>`, `extract_match.py`, `train_touch.py` (fold commands in
`$MTAH_WORK/video_touch/PAUSED_GPU.md`), `eval_touch.py runs/cv`, `st_phase3.py runs/cv`,
`st_spell_diag.py runs/cv`, `match_remove.py`. Data, crops (114 GB) and weights stay out of the repo.
