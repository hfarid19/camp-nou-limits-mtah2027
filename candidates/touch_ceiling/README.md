# touch_ceiling: how good would a video touch detector have to be to beat ensemble_ctrl_v6?

CPU-only simulation on the 48 PFF World Cup 2022 selection games (no video, no GPU, no holdout
games). Take PFF's true touches, corrupt them with a detector model (recall, precision, timing,
team), build possession labels from them alone or combined with v6, and score against the PFF
touch labels.

**Bottom line.**
- The real detector (`../video_touch`) loses badly in every setting. At its operating point the
  best combination is −0.075 spellF1 vs v6 (0/48 games), and touch-only labels score 0.20–0.22
  against v6's 0.76. The only setting that doesn't lose, C1s, also doesn't change v6 (≤ 0.004
  either way).
- To beat v6, a detector needs:
  - team accuracy ≥ 0.95;
  - recall and precision ≥ 0.9 (≥ 0.8 with perfect team);
  - recall ≥ 0.5–0.7 on the touches that change the team (interceptions and tackles), even when
    everything else is perfect.
- Timing barely matters: σ up to 5 frames is fine, and σ up to 10–15 frames is fine when the
  other knobs are perfect.
- The best combination with v6 (C2: use touches only where v6's members disagree) reaches
  break-even at about r = p = 0.9, a = 0.95, σ ≤ 5 (r_chg = r). Its ceiling is only +0.05.
- The largest possible gain is +0.10 (C123), but it needs p and a ≥ 0.95.

## Method

- **Truth touches** (`touches.py`). These are the on-the-ball game events (OTB and kick-offs with
  `home_team`) that `../pff/labels.py`'s touch reading uses. Each event maps to the first 25 fps
  graded row whose PFF frameNum is ≥ its `start_frame`. Dead = npz status 0 (OUT → restart).
  - Count: 60,317 touches in 48 games. 4,701 are restarts or kick-offs, 12,305 (20%) change the
    team in open play (interceptions, tackles, won duels, and deflections that flip PFF's touch
    team), and 43,311 keep the team.
  - Check: touches plus dead rebuild the bench's touch label **identically on every row of all 48
    games** (spellF1 1.0000 with both `possession_metrics.spell_scores` and our fast copy).
- **Detector model** (`sim.py`). Common random numbers across configs, so the surfaces are monotone.
  - `r`: recall on restart and same-team touches.
  - `r_chg`: recall on team-changing touches (= r unless set).
  - `p`: expected precision. We add `round(n_detected·(1−p)/p)` false touches at uniform random
    truth-alive rows; each is the opponent's w.p. 0.5.
  - `σ`: Gaussian timing jitter in frames.
  - `a`: a detected true touch keeps its team w.p. a.
  - Real detector point: r 0.43, r_chg 0.2, p 0.475, a 0.75, σ 4. Its ±5-frame P/R is
    0.475/0.431. 58% of its ±5 matches fall within ±2 frames, which Gaussian σ ≈ 4 reproduces.
    Tackle and block recall is 10–25% and team accuracy 72–78%.
- **v6** = the shipped rule on the members' out-of-fold PFF codes (`probs/nn_pff/{t5c,m5c,radar_pff}`,
  `vote_offline.vote`, merge_short 1 s + control.fold 1.25 s; prekick_dead is a no-op without tags).
  - On the 48 games: **touch spellF1 0.7616, fine3 0.7815.** This matches the README's ~0.78.
  - It is inflated because PFF's ball is event-placed. 75% of v6's H↔A switches fall within 5
    frames of a true switch to the same team.
- **Settings.** Team on alive frames:

| setting | dead ball | team |
|---|---|---|
| A | truth | last detected touch (upper bound) |
| B | v6 | last detected touch on a v6-alive row of the current v6-alive run; v6's team before the run's first detected touch |
| C1_k (k 5/12/25) | v6 | v6, but each H↔A switch is kept only if a detected touch of the new team lies within k frames; it is then moved to that touch. Unconfirmed switches are dropped. |
| C1s_k | v6 | same, but unconfirmed switches are kept (retime only) |
| C2 | v6 | v6 where its 3 members agree; where they disagree (7.3% of v6-alive frames, holding 48% of v6's wrong-team frames), team from B |
| C3r | v6 | a v6 spell < 3 s (not the first of its alive run) is kept only if a detected touch of its team lies inside it (±5 frames); else it takes the previous team |
| C3 | v6 | C3r + insertion: a detected opponent touch inside a v6 spell opens an opponent spell until the next detected touch of the spell's team, if that is < 3 s later |
| C13 | v6 | C1s_12, then C3 |
| C123 | v6 | C13, then C2's override where the members disagree |

- **Scores.** Mean over the 48 games of seed-averaged spellF1 vs the touch labels (raw), and fine3
  (mean of raw, control.fold T0.5 and T1). "Beats v6" = mean paired difference > 0. SEs are about
  0.002–0.006 over games, and every headline difference is many SE from 0.
- **Runs.**
  - Grid: r, p ∈ {0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1}, a ∈ {0.75, 0.85, 0.95, 1}, σ ∈ {0, 2, 5},
    r_chg ∈ {= r, 0.2}; 2 seeds; 15 min on 4 CPU processes.
  - Real neighbourhood, one-knob sweeps and r_chg × r sweeps: 1,052 configs, 2 seeds, 13 min.
- Numbers below are raw spellF1 unless marked; fine3 gives the same conclusions (it is in the
  regenerated report).

## Perfect touches and the real detector (raw spellF1, diff vs v6 0.7616)

| | A | B | C1_25 | C1s_12 | C2 | C3r | C3 | C13 | C123 |
|---|---|---|---|---|---|---|---|---|---|
| perfect | 1.000 (+0.238) | 0.919 (+0.158) | 0.752 (−0.009) | 0.763 (+0.001) | 0.812 (+0.050) | 0.772 (+0.010) | 0.862 (+0.100) | 0.864 (+0.103) | **0.866 (+0.104)** |
| **real detector** | 0.204 (−0.558) | 0.224 (−0.538) | 0.349 (−0.412) | 0.760 (−0.001) | **0.686 (−0.075)** | 0.691 (−0.071) | 0.468 (−0.294) | 0.470 (−0.292) | 0.444 (−0.318) |

- fine3 (v6 0.7815): at the real point A 0.205, B 0.226, C2 0.700, C3r 0.713, C123 0.449; with
  perfect touches A 0.972, B 0.901, C2 0.819, C123 0.852.
- At the real point, every setting loses on 0/48 games, apart from C1s, which is v6 ± 0.001.
- C1_k with k 5/12/25 (drop unconfirmed switches) loses even with perfect touches: −0.072,
  −0.044, −0.009. v6 sometimes places a correct switch more than k frames from the touch (e.g.
  48 frames early on a 1000-frame spell), and dropping it erases the whole spell.
- C1s (retime only) can't help, because v6's PFF switches are already where the touches are.
- Gains have to come from fixing v6's team (C2) or its short spells (C3*).
- B's ceiling (0.919) shows v6's dead ball costs 0.08 even with perfect touches.

## Break-even

**One knob at a time, the others perfect** (σ 0; worst value that still beats v6; r_chg = r for the r column):

| setting | r (with r_chg) | r_chg only | p | a | σ (frames) |
|---|---|---|---|---|---|
| A | ≥ 0.7 | ≥ 0.6 | ≥ 0.9 | ≥ 0.95 | ≤ 15 |
| B | ≥ 0.8 | ≥ 0.7 | ≥ 0.95 | ≥ 0.95 | ≤ 15 |
| C2 | ≥ 0.6 | ≥ 0.5 | ≥ 0.8 | ≥ 0.9 | ≤ 15 |
| C3r | ≥ 0.9 | ≥ 0.9 | any | ≥ 0.95 | ≤ 4 |
| C13 | ≥ 0.6 | ≥ 0.6 | ≥ 0.95 | ≥ 0.95 | ≤ 15 |
| C123 | ≥ 0.7 | ≥ 0.6 | ≥ 0.95 | ≥ 0.95 | ≤ 10 |

**Uniform quality r = p = q** (r_chg = r): smallest q that beats v6 (raw; '–' = not even q = 1).
No setting beats v6 at a ≤ 0.85 (C1s ties).

| setting | a 0.95 σ 0 | a 0.95 σ 2 | a 0.95 σ 5 | a 1 σ 0 | a 1 σ 2 | a 1 σ 5 |
|---|---|---|---|---|---|---|
| A | 0.98 | 0.98 | 1 | 0.9 | 0.95 | 0.95 |
| B | 1 | – | – | 0.95 | 0.95 | 0.95 |
| **C2** | **0.9** | **0.9** | **0.9** | **0.8** | **0.8** | **0.9** |
| C3r | 0.95 | 0.98 | – | 0.9 | 0.95 | – |
| C13 / C123 | 1 | 1 | – | 0.95 | 0.95 | 0.95 |

The same table against the **honest bar 0.729** (v6's DFL fine4 / Metrica scores), for A and B
only, because C* inherit v6's inflated PFF codes:
- A needs q ≥ 0.95–0.98 at a = 0.95, or q ≥ 0.9 at a = 1.
- B needs q ≥ 0.98–1 at a = 0.95, or q ≥ 0.95 at a = 1.
- Nothing reaches it at a ≤ 0.85.

**Smallest r per p** (grid, r_chg = r): cell = smallest r that beats v6.

| a, σ | A: p 0.9 | A: p 0.95 | A: p 1 | B: p 0.95 | B: p 1 | C2: p 0.8 | C2: p 0.9 | C2: p 1 | C13: p 0.95 | C13: p 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.95, 0 | – | 0.98 | 0.9 | – | 1 | 0.95 | 0.8 | 0.7 | – | 0.9 |
| 0.95, 2 | – | – | 0.95 | – | – | – | 0.8 | 0.7 | – | 0.9 |
| 0.95, 5 | – | – | 0.98 | – | – | – | 0.9 | 0.7 | – | – |
| 1, 0 | 0.9 | 0.8 | 0.7 | 0.9 | 0.8 | 0.8 | 0.7 | 0.7 | 0.8 | 0.7 |
| 1, 2 | 0.95 | 0.8 | 0.8 | 0.9 | 0.8 | 0.8 | 0.7 | 0.7 | 0.8 | 0.7 |
| 1, 5 | 1 | 0.9 | 0.8 | 0.9 | 0.8 | 0.9 | 0.7 | 0.7 | 0.9 | 0.7 |

With **r_chg = 0.2** (the real detector's interception and tackle recall), **no setting beats v6
anywhere in the grid**, even at r = p = a = 1 and σ = 0.

**r_chg × r** (σ 4), diff vs v6:

| C2, p 1, a 1 | r_chg 0 | 0.2 | 0.5 | 0.7 | 0.9 | 1 |
|---|---|---|---|---|---|---|
| r 0.43 | −0.034 | −0.023 | −0.004 | +0.011 | +0.029 | +0.039 |
| r 1 | −0.030 | −0.017 | +0.001 | +0.015 | +0.031 | +0.039 |
| **C2, p 0.8, a 0.95** | | | | | | |
| r 0.43 | −0.045 | −0.038 | −0.025 | −0.014 | −0.002 | +0.004 |
| r 1 | −0.050 | −0.041 | −0.030 | −0.020 | −0.010 | −0.005 |
| **B, p 1, a 1** | | | | | | |
| r 0.43 | −0.286 | −0.231 | −0.123 | −0.035 | +0.067 | +0.122 |
| r 1 | −0.250 | −0.187 | −0.082 | −0.005 | +0.080 | +0.125 |
| **C123, p 1, a 1** | | | | | | |
| r 1 | −0.105 | −0.076 | −0.032 | +0.005 | +0.047 | +0.070 |

## Which knob matters most

Score change for one step away from perfect, and the gain at the real point from fixing one knob:

| setting | r 1→0.9 | r_chg 1→0.7 | p 1→0.9 | a 1→0.95 | σ 0→4 | real point | r, r_chg→1 | p→1 | a→1 | σ→0 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | −0.068 | −0.158 | −0.177 | −0.162 | −0.068 | 0.204 | −0.038 | +0.087 | +0.069 | +0.003 |
| B | −0.051 | −0.143 | −0.164 | −0.152 | −0.033 | 0.224 | −0.075 | +0.140 | +0.067 | −0.002 |
| C2 | −0.010 | −0.028 | −0.016 | −0.018 | −0.011 | 0.686 | −0.032 | +0.026 | +0.019 | −0.002 |
| C123 | −0.027 | −0.077 | −0.119 | −0.088 | −0.035 | 0.444 | −0.234 | +0.158 | +0.033 | +0.001 |

- **Team accuracy is the most sensitive knob per unit**: 5 points cost 0.15–0.16 in A/B.
  **Precision comes next**: 10 points cost 0.16–0.18.
  - Every wrong-team touch, true or false, opens a spurious spell and splits a true one.
  - At the real point, more recall makes things worse (r, r_chg → 1: −0.04 to −0.23). With
    p 0.475 and a 0.75, more touches mean more wrong switches.
- **Recall on team-changing touches is the hard floor.** Even with everything else perfect:
  - A needs r_chg ≥ 0.6, B ≥ 0.7, and C2 ≥ 0.5.
  - At realistic p and a, C2 needs r_chg ≈ 1.
  - The real detector is at 0.1–0.25.
- **Timing is the least sensitive.** spellF1's IoU matching tolerates σ ≤ 10–15 frames when the
  other knobs are perfect, and σ 5 barely moves the break-even.

## Could a SoccerNet-level spotter reach this?

Not plausibly. SoccerNet ball-action spotters at roughly 0.6–0.8 mAP with tight tolerance work at
around P ≈ R ≈ 0.7–0.8 on their best classes. Their weakest classes are the rare, contested ones,
which are exactly the team-changing touches. Adding team attribution lowers the scores further.
Break-even needs P and R ≥ 0.9, team ≥ 0.95, and tackle/interception recall ≥ 0.5–0.7 at the same
time. Even then the best hybrid gains only +0.01 to +0.05 on PFF, where v6's score is itself
inflated.

## Files and reproduce

```
export MTAH_WORK=...; P=$MTAH_WORK/venv/bin/python; export OMP_NUM_THREADS=2
$P candidates/touch_ceiling/touches.py                   # step 1 + exact-rebuild check -> arch/touch_ceiling/touches.pkl
$P candidates/touch_ceiling/sim.py smoke                 # 3 games, 3 configs
$P candidates/touch_ceiling/sim.py grid --jobs 4         # -> grid.pkl (15 min)
$P candidates/touch_ceiling/sim.py real --jobs 4 --seeds 2   # -> real.pkl (13 min)
$P candidates/touch_ceiling/report.py --metric raw       # every table (also --metric fine3, --full)
```

- `common.py`: paths, games and targets (vote_offline's cached `pff_targets.pkl`). Also a fast
  spellF1 (vectorised overlap search, same greedy matching), equal to `pm.spell_scores` to the last
  bit on v6 and on perturbed codes. Also the v6 OOF codes, cached in `arch/touch_ceiling/v6_codes/`.
- Outputs (not in the repo) are in `$MTAH_WORK/arch/touch_ceiling/`: `touches.pkl`, `grid.pkl`,
  `real.pkl`, `report_raw.md`, `report_fine3.md`, and the logs.
- Not used: `evaluation/gt`, `evaluation/reports`, `run_evaluation.py`, the 16 PFF holdout games,
  BAR-ATH labels, torch, the GPU.
