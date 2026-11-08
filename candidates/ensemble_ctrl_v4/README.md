# ensemble_ctrl_v4: the vote retuned to the ground truth's possession granularity (submission)

`pipeline.py` does four things in order:

1. A hard majority vote of `temporal_tcn_v2` and `handover_mstcn_v2`, each in its **raw-flag
   training variant** (`pipeline_raw.py`: trained on the DFL BallPossession flag as it is,
   without folding short spells), plus `ball_radar`, in that tie-break order.
2. `learned_dfl_evidence`'s `merge_short_v2` at 1.0 s, as in every earlier vote.
3. `control.fold(codes, 1.25 s)`: a predicted opponent spell shorter than 1.25 s that directly
   follows the other team's possession is handed back.
4. `prekick_dead`: every graded frame before a half's kick-off is D. FAQ.md states that the second
   half is graded from frame 110,808 but kicks off at 113,742. The kick is placed 88 frames after
   the start of that half's "Kick Off (Start)" Smart Tagging instance. That lag comes from the
   first half, which is graded from the kick itself (tag start 11,097, kick 11,185). On the match
   this gives D up to frame 113,748. The previous submission had `A` from 113,664: 78 frames that
   were certainly wrong, plus a spurious possession. On the DFL and Metrica benches there are no
   tags, so it does nothing there.

`members.py` and `ensemble_core.py` are copies of `ensemble_ctrl_v2`'s. The only change is that
`members.load` takes a pipeline file name, so a member's training variant can be loaded.

## Why the change from ensemble_ctrl_v2

`ensemble_ctrl_v2` bet that the competition's "controlled possession" means a coarse,
multi-second control (members trained on 2 s-folded targets, a 2 s output fold). The organisers'
own numbers say otherwise.

EVALUATION.md lists the full-match spellF1 of a perfect submission shifted 3 / 8 / 25 frames
late: 0.9818 / 0.8933 / 0.6539. These cannot be the 17-possession public window, where every
score is a multiple of 1/85. Under a uniform shift, a spell survives an IoU threshold t only if
(L − k) / (L + k) ≥ t. So the three numbers are a fingerprint of the truth's spell lengths.

| labels | 3 frames late | 8 frames late | 25 frames late |
|---|---|---|---|
| **competition truth (EVALUATION.md)** | **0.982** | **0.893** | **0.654** |
| DFL raw flag | 0.960 | 0.862 | 0.557 |
| DFL folded at 0.5 s | 0.995 | 0.910 | 0.597 |
| DFL folded at 1 s | 0.998 | 0.958 | 0.663 |
| DFL folded at 2 s (v2's target) | 0.999 | 0.989 | 0.797 |
| DFL event labels, HOLD 3 s | 0.996 | 0.973 | 0.798 |
| Metrica practice labels (Sample Games 1, 2) | 0.94 | 0.88 | 0.66–0.67 |

The truth lies between the raw DFL flag and a 1 s fold, and it looks like Metrica's own
event-based possession. It does not look like the 2 s reading.

The selection metric was therefore changed, before any candidate was scored on it, to **fine4**
(`candidates/score_sel2.py`). fine4 is the mean spellF1 against the raw DFL flag, fold
T = 0.5 s, fold T = 1 s, and the event labels with HOLD 1 s.

## Selection

Everything was selected **nested** (pick on 6 DFL games, score the 7th) over this grid:
- member training variants: TCN ctrl / raw / ctrl_mix × MS-TCN ctrl / raw, plus the v1 members;
- vote smoothing: off or 1 s;
- output fold: 0 to 2 s.

The search is `candidates/_scratch/nested_fine.py`, offline, on saved leave-one-game-out codes.
It picked raw / raw, 1 s smoothing, fold 1.25 s in 6 of 7 folds; the seventh pick differs only
by fold 0.

| nested, vs ensemble_ctrl_v2 (7 games × 3 seeds) | difference | games won |
|---|---|---|
| fine4 | **+0.047 ± 0.002** | 7/7 |
| raw flag / fold 0.5 / fold 1 / event HOLD 1 | +0.051 / +0.048 / +0.042 / +0.046 | 7/7 each |
| sel2 (the 2026-11-08 metric, centred on T ≈ 1.5–2 s) | −0.010 ± 0.005 | 2/7 |
| fold 2 s | −0.040 ± 0.005 | 0/7 |

This is a deliberate bet on the fine reading, which the published fingerprint supports. If the
truth were the 2 s reading after all, the bet would cost about 0.04.

## Results (live runs)

| | fine4 | sel2 | ctrl_mean | raw spellF1 | macroF1 | Metrica dev | Metrica holdout | match possessions |
|---|---|---|---|---|---|---|---|---|
| **ensemble_ctrl_v4** | **0.7095** | 0.7409 | 0.7193 | **0.6737** | **0.9272** | **0.7084** | **0.7071** | 322 |
| ensemble_ctrl_v2 | 0.6619 | 0.7485 | 0.7443 | 0.6223 | 0.9227 | 0.6643 | 0.6780 | 262 |

- **DFL bench v2:** the live run reproduces the offline codes on all 21 game-seeds. fine4 gains
  +0.048 ± 0.001 and wins all 21 game-seeds.
- **Metrica dev:** Sample Games 1 and 2, with practice labels from Metrica's own events. That
  labelling style matches the fingerprint.
- **Metrica holdout:** Sample Game 3, run once for this candidate.
- **Match:** the CSV is deterministic (sha256 `2ea055d4…`, two runs `cmp`-identical), valid, and
  has dead share 0.476. 17 possessions in the public 5 minutes would be about 306 per 90 min; the
  public labels were not used.

Reproduce:

```
OMP_NUM_THREADS=2 python candidates/bench.py candidates/ensemble_ctrl_v4/pipeline.py --dfl2 \
    --folds $MTAH_WORK/arch/ensemble_ctrl_v2/folds.json --save-probs $MTAH_WORK/probs/ensemble_ctrl_v4
python candidates/score_sel2.py $MTAH_WORK/probs/ensemble_ctrl_v4 --ref $MTAH_WORK/probs/ensemble_ctrl_v2
python candidates/bench.py candidates/ensemble_ctrl_v4/pipeline.py --match
```

The folds JSON already maps the raw-variant weight packs (`tcn_weights_raw.npz`,
`model_tcn_raw.npz`) to their leave-one-game-out fold models in `$MTAH_WORK/arch/*_v2/raw/`.
