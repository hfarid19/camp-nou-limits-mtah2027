# skillcorner_domain

**Question.** learned_dfl_evidence (LDE) trains its LightGBM frame models only on the 7 open
DFL/IDSSE games. Does SkillCorner's open broadcast tracking help as a second training domain?
Broadcast tracking is noisy and partial, like the competition's automatic tracking.

**Short answer: no on DFL. Mixed on Metrica.** Adding SkillCorner to the **home/away** model makes
under-splitting much worse: DFL LOGO spellF1 0.6243 -> 0.5691, and match possessions 325 -> 298.
Adding it to the **dead/alive** model costs 0.006 on DFL (0.6185, inside the noise band) and gains
0.0125 on Metrica dev (0.6725 -> 0.6850). The shipped variant is that last one (v2). It does not
beat LDE on the primary selection set (DFL LOGO). Its value is a dead model trained on a
different domain, so it gives the ensemble a somewhat different view.

## Method

1. `fetch_skillcorner.py` downloads github.com/SkillCorner/opendata at the pinned commit
   `4340d274572876239c154c90bc507a9b3250a656`. That is 20 Australian A-League 2024/25 matches:
   tracking jsonl through git-LFS, match json, Phases of Play and Dynamic Events. Every tracking
   file is checked against the sha256 and size in its LFS pointer, and `manifest.json` records
   the hash of every file. The download is about 1.8 GB.
2. `parse_skillcorner.py` writes the files in the DFL npz layout at 25 fps.
   - **Labels (algorithmic, no smoothing, nothing hand-made).** Inside `match_periods`, a frame
     is D where the tracking frame's `possession.group` is null. Otherwise it is H or A from
     `group` ("home team" / "away team"). Frames are resampled 10 -> 25 fps with the nearest
     source frame.
   - **Why this matches the competition's meaning.** `group` is a team possession. It persists
     through passes and loose balls, and it is null while the ball is out of play. On the 2
     games checked it agrees with the Phases of Play windows, which exist only while the ball is
     in play, on about 95% of frames. The ball is outside the pitch on 0.1-0.4% of group frames
     but on 12-16% of null frames. That is the same meaning as DFL's BallStatus / BallPossession
     and the GT definition: D where the tracking marks the ball dead, otherwise the team it
     names.
   - **Positions.** Coordinates are moved from the centre origin and scaled to 105 x 68. Player
     positions are linearly interpolated, but only between two frames where the player is
     detected.
3. `make_dataset_sc.py` builds the features.
   - **Players.** It uses SkillCorner's extrapolated positions for all 22 players: the
     competition tracks about 20.5 objects, while detected-only broadcast gives about 9.
   - **Ball.** Only the detected ball is used. It is missing on 40% of graded frames, 7-8% while
     alive and about 80% in stoppages (replays, close-ups). That is very close to the
     competition's 41% missing, concentrated in stoppages, without any synthetic blackout.
   - **Degradation.** `tune_on_sample_data.degrade`: anonymous fragmented slots, tracklet mean
     U(30, 100) s, target missing U(0.33, 0.48), plus ghost duplicates as in LDE. Seed 21,
     stride 5, about 591k rows over 20 games.
   - **Features.** LDE's `features.compute`, unchanged (194 features).
4. `train_sc.py` trains LDE's tasks and parameters: d (dead vs alive), ha (home vs away with
   team-swap augmentation) and c. Training uses the LDE DFL datasets (`make_dataset.py`, seeds
   11-13; r1 for ha, r2 for d) plus all SkillCorner rows at weight 1. **LOGO stays on DFL:** fold
   gi trains on the other 6 DFL games plus all 20 SkillCorner games, and its log-loss curve is
   scored on DFL game gi. The full model's rounds are the argmin of the mean held-out DFL curve.
5. The decoder and inference code in `pipeline.py` are LDE round 3, unchanged: Viterbi, local
   contrast, gap snaps, hand-over peak snap. Only these were added:
   - `predict_proba`: the decoder's own per-frame evidence, P(H) = ph(1-pd), P(A) = (1-ph)(1-pd),
     P(D) = pd, after the long-gap term, the contrast and the tag priors.
   - a feature cache, so predict and predict_proba compute the features once.
   - an `SCD_` sweep prefix.

## SkillCorner label statistics (20 games, 1,971 min)

| | SkillCorner (this) | DFL (LDE README) |
|---|---|---|
| possession spells per 90 min | 305 | 412 |
| dead share | 0.459 | 0.448 |
| spell duration p50 | ~5.6 s (per-game 4.8-7.1) | 3.7 s |
| spells < 1 s | ~10.8% | 12.5% |
| ball missing (graded) | 0.40 | - (degraded) |

SkillCorner's team possession is stickier than DFL's flag: about 25% fewer spells, with longer
medians. This explains the home/away result below. Per-game numbers are in
`<work>/npz/label_stats.json`.

## Variants (all numbers from `candidates/bench.py` runs actually executed)

DFL = `--dfl` LOGO (7 games, seed 2027, fold models). Dev = Metrica Games 1-2 x 2 seeds. Match =
possessions in the match CSV (no score). The DFL possession totals are summed over the 7 games
(true total 3061).

| variant | d / ha / c | held-out DFL log-loss | DFL spellF1 / macroF1 | DFL poss pred | dev spellF1 / macroF1 | match poss |
|---|---|---|---|---|---|---|
| v0 = LDE (reproduced) | LDE / LDE / LDE | d 0.0586, ha 0.2754 | **0.6243** / 0.9144 | 2202 | 0.6725 / 0.8145 | 325 |
| v1 | LDE / **+SC** / LDE | ha 0.2731 (600 rounds, cap) | 0.5691 / 0.9069 | 1921 | 0.6083 / 0.8068 | 298 |
| v1, switch_cost 0.4 | same | | 0.5716 / 0.9071 | | | |
| v1, switch_cost 0.2 | same | | 0.5735 / 0.9073 | 1989 | | |
| **v2 (shipped)** | **+SC** / LDE / LDE | d 0.0652 (700 rounds, cap) | 0.6185 / 0.9120 | 2143 | **0.6850** / 0.8168 | 314 |
| v3 | +SC / +SC / LDE | | 0.5603 / 0.9043 | 1883 | | |

Per-game DFL spellF1 for v2: J03WMX 0.5496 (307/511), J03WN1 0.5692 (306/500), J03WOH 0.6592
(291/402), J03WOY 0.6112 (311/456), J03WPY 0.6318 (295/421), J03WQQ 0.6473 (343/401), J03WR9
0.6612 (290/370).

What the numbers say:
- **Home/away (v1).** Held-out DFL log-loss even improves slightly (0.2754 -> 0.2731), but the
  model learns SkillCorner's stickier possession. Its team probability stops flipping on the
  short touches that DFL's flag (and the competition GT) count. Predicted possessions fall on
  every DFL game (2202 -> 1921), on dev, and on the match (325 -> 298). That makes the main known
  weakness, under-splitting, worse. A lower switch cost does not recover it (0.5716 / 0.5735):
  the evidence itself is smoother. This part of the architecture does not work here.
- **Dead/alive (v2).** Held-out DFL log-loss gets worse (0.0586 -> 0.0652). DFL spellF1 drops
  0.006, within the ~0.04 noise band but in the wrong direction. Metrica dev rises 0.0125, on all
  4 rows (+0.008 to +0.015), and macroF1 rises 0.002. One likely reason: SkillCorner's stoppages
  look like real automatic tracking (the ball and many players vanish during replays), while
  DFL's are synthetic blackouts. The match prediction has fewer possessions (314) and the same
  dead share (0.477 vs 0.472).
- **Hand-over model c.** Not retrained with SkillCorner. The 40-minute training budget went to
  ha (~16 min) and d (~18 min). Since ha shows that SkillCorner's possession changes are
  stickier, a c model trained on its flips would most likely blur the snap peaks too.
- **Untried for lack of budget.** A down-weighted SkillCorner home/away mix (w_sc < 1). It would
  most likely land between v1 and v0. Detected-only players (`--players det`): the dataset code
  supports it, but no model was trained on it.

## Final scores (v2, shipped)

| split | spellF1 | macroF1 |
|---|---|---|
| DFL LOGO | 0.6185 | 0.9120 |
| Metrica dev | 0.6850 | 0.8168 |
| Metrica holdout G3 (run once, at the end) | 0.6832 | 0.8526 |

On the match: 314 possessions, mean 10.6 s, dead share 0.477, no X in the halves. Two `--match`
runs give byte-identical `submission.csv` and `match.npy` (checked with `cmp`). The probabilities
of every scored game are in `<scratch>/probs/skillcorner_domain/`.

## Data sources and licences

- **SkillCorner Open Data**, github.com/SkillCorner/opendata, commit 4340d27. MIT License,
  Copyright (c) 2020 SkillCorner. SkillCorner asks users to credit them. The labels come from
  SkillCorner's own tracking product: `possession.group`, produced by their automatic pipeline.
  We did not hand-label or edit anything, and none of it is about the competition match.
- **DFL / IDSSE open data** (through learned_dfl_evidence): Bassek et al., Sci Data 2025,
  figshare 28196177 v1, CC BY 4.0.
- Raw data and datasets live outside the repo, in the scratch work dir. Nothing derived from the
  match is stored here except `submission.csv`.

## Files

- `fetch_skillcorner.py`, `parse_skillcorner.py`, `make_dataset_sc.py`, `train_sc.py`: the
  SkillCorner chain described above.
- `features.py`, `make_dataset.py`: LDE copies, unchanged. The feature list must match the
  models.
- `pipeline.py`: LDE round-3 pipeline plus `predict_proba`.
- `model_d.txt`: trained by train_sc.py. `model_ha.txt` and `model_c.txt`: LDE's files,
  byte-identical.
- `model_meta.json`: feature names, lgb version, sha256 of each model (asserted at load) and the
  per-model provenance. `model_meta_lde.json`: LDE's original meta.
- `install_variant.py`: the switcher used for this study. It copies a variant's models and
  writes its meta and folds json. Its paths are hard-coded to the scratch work dir.
- `rebuild_model_d.sh`: the rebuild.

## Rebuild

    pip install -r candidates/skillcorner_domain/requirements.txt
    # DFL round-2 dataset as in candidates/learned_dfl_evidence/rebuild_models.sh, then:
    sh candidates/skillcorner_domain/rebuild_model_d.sh WORK_DIR DFL_R2_CACHE

That runs fetch_skillcorner -> parse_skillcorner -> make_dataset_sc (seed 21, players ex) ->
`train_sc.py --models d --max-rounds 700`, and ends with `cmp` against `model_d.txt`. Training is
deterministic (deterministic=True, fixed seeds and threads), but this rebuild has **not** been run
a second time to confirm byte identity, because of the time budget. The DFL dataset used here was
`<scratch>/idsse_r2d/ds`, whose config.json matches make_dataset.py's defaults.

Bench (the folds json maps model sha256 prefixes to the fold dirs):

    python candidates/bench.py candidates/skillcorner_domain/pipeline.py --dfl --folds <scratch>/arch/skillcorner_domain/folds.json
    python candidates/bench.py candidates/skillcorner_domain/pipeline.py [--match | --holdout]
