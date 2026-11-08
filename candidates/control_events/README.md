# Event-defined "controlled possession" labels for the 7 open DFL games

The bench scores against DFL's tracking flag BallPossession, which flips on every touch.
CHALLENGE.md counts a change of possession only when the opponent gains **controlled** possession.
Until now we stood in for that with `control.fold(T)` and averaged over T = 1, 2, 3 s
(`ctrl_mean`). Here the DFL **event** data defines control instead, and our model choices are
re-checked against those labels.

## Data and licence

Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", *Scientific Data* 12:195. figshare article 28196177 v1
(doi 10.6084/m9.figshare.28196177.v1). The data are from DFL Deutsche Fussball Liga, licence
**CC BY 4.0**. The files used are the 7 `DFL_03_02_events_raw_*.xml` (0.6–0.8 MB each), plus the
positions and matchinformation XMLs already in `$MTAH_WORK/idsse/raw`.

| script | what it does |
|---|---|
| `fetch_events.py` | Downloads only the 7 event XMLs into `$MTAH_WORK/idsse/events/`. Figshare ids, sizes and md5s are pinned in the script. A file whose md5 does not match is deleted. |
| `frametimes.py` | Reads the ball FrameSets' `(N, T)` from the positions XMLs into `idsse/events/<match>_frametimes.npz`. The rows are identical to `idsse/npz/<match>.npz`: 25 fps with no gaps, and one jump at half time. |
| `events.py` | Parses the events (regex, no DOM; `Delete` placeholders are dropped), derives control actions, and aligns events to rows. |
| `align_check.py` | Alignment diagnostics. Output in `$MTAH_WORK/arch/control_events/align_check.txt`. |
| `labels.py` | Writes `$MTAH_WORK/ctrl_events/<match>.npy`: int8 per raw DFL frame, codes H 0 / A 1 / D 2 / X 3, with the same rows as `gt` in `bench.dfl2_prepared`. |
| `score_events.py` | Scores `probs/<slug>/dfl2_<match>_<seed>_codes.npy` against the labels with `possession_metrics.evaluate`. |
| `analyse.py`, `robust_T.py` | All the tables below. Outputs are in `$MTAH_WORK/arch/control_events/`, with label variants under `variants/`. |

```
MTAH_WORK=... python -I candidates/control_events/fetch_events.py
python -I candidates/control_events/frametimes.py      # ~1 min, reads the 7 positions XMLs once
python -I candidates/control_events/labels.py
python -I candidates/control_events/score_events.py $MTAH_WORK/probs/ensemble_vote ...
python -I candidates/control_events/analyse.py
```

As a check, `score_events.py` reproduces the bench's numbers for ensemble_vote: raw 0.6564 and
ctrl_mean 0.6961.

## Alignment (events to tracking rows)

`EventTime` is a live wall-clock stamp, and it is **not** synchronised with the positions:

- **Per-game clock offset.** The offset is −2.0 to +1.3 s, depending on the game. Before
  alignment, the acting player is within 1.5 m of the ball at the event row for only 31% of
  open-play player events.
- **Per-event jitter of about 1 s.** On top of that, about half of all shots are logged 2–10 s
  late. DFL itself supplies a tracking-based `CalculatedFrame` for shots, and it confirms this.
- **The flags are set live by an operator.** BallStatus and BallPossession lag the geometry
  slightly.

The alignment (`events.align`, `events.flag_lag`) works in four steps:

1. An event with a `CalculatedFrame` takes that frame.
2. Every other event is shifted by one **per-game offset** from the **positions only**: the shift
   (±4 s) that maximises the share of open-play player events whose player is within 1.5 m of the
   ball.
3. An event with a tracked acting player snaps to the row within ±1.5 s where that player is
   closest to the ball, provided that distance is ≤ 2 m. This applies to 79–93% of player events.
4. To compare events with the operator flag, one **per-game flag lag** is added. It is the shift
   in [−1, 2] s that maximises agreement between the event team and the flag: +0.12 s in 5 games,
   +0.16 s and +0.40 s in the other two. It is a single constant per game, so it cannot move
   individual events into individual spells.

Results (all 7 games, `align_check.txt`):

| | player within 1.5 m of the ball | set piece within 1 s of a BallStatus 0→1 switch | shots within 0.5 s of CalculatedFrame | event team = flag team (open play) |
|---|---|---|---|---|
| raw EventTime | 0.31 | 0.34 | 0.24 | 0.875 |
| aligned (positions clock) | 0.87 | 0.66 | 1.00 (by construction) | 0.898 |
| + flag lag (flag clock) | — | 0.67 | — | 0.936 |

Kick-offs: the second-half and after-goal kick-offs land on the BallStatus switch exactly
(for example in J03WMX, rows 13016, 70708 and 125641). The first-half kick-off is the first
tracking row, because tracking starts at the kick. One after-goal kick-off in J03WMX is 7 s off a
switch; the cause is probably a VAR delay. Set pieces stay loose (0.66 within 1 s), because the
taker stands at the ball before the kick and the snap can pick a frame during that wait. This does
not matter for the labels, because spells after a stoppage are never handed back. The residual
error of open-play events is what the ±0.5 s tolerance below absorbs. Varying the tolerance or
using only snapped events barely moves anything (see Robustness).

## Rules

These rules are based on the provider definitions in `mtah-work/research/possession_tracking.md`:

- Opta: a defensive action starts a possession only if a controlled action follows.
- Metrica: an opponent's touch does not end possession if the play still reaches its target.
- The DFL operator brief: possession needs "a touch with ball control".
- The challenge's own proxy: a player keeps the ball for a few seconds.

The rules start from the DFL BallPossession spells and do not change any boundary. They only hand
spells back.

- **R0. Dead ball.** Frames with BallStatus 0 stay D. Frames with no flag stay X. Exactly as in
  `bench.dfl2_prepared`.
- **R1. Restarts.** A spell that starts a run (after D, or at the start of a half) is always kept.
  CHALLENGE.md: "after a restart there is always a clear possessor". This is the same rule as in
  `control.fold`.
- **R2. Contested spells.** A DFL spell of team X that directly follows possession of the other
  team Y is a change of possession only if one of these holds:
  - **C1.** X has a **controlled action** whose aligned frame (flag clock) lies within the spell
    ± **TOL = 0.5 s**. Controlled actions are:
    - a `Play` (pass or cross, successful or not), including the play inside a set piece;
    - a `ShotAtGoal`;
    - a `BallClaiming` (InterceptedBall, BallClaimed, BallHeld, Block);
    - a `TacklingGame` won by X with `WinnerRole = withoutBallControl` and
      `PossessionChange = true` (a won tackle that changes possession);
    - a `TacklingGame` in which an X player has the role `withBallControl`, as winner or loser.
      This is DFL's own statement that the player had the ball under control. It includes
      dribbles (`dribbledAround`, `ballControlRetained`).
    - `Run` and `Nutmeg` (dribbles or carries).
  - **C2.** The spell lasts at least **HOLD = 3.0 s** (the "holds the ball for a few seconds"
    proxy). It is needed because DFL logs no receptions and almost no carries: a player who
    receives the ball and dribbles for 10 s often has no event at all.

  If neither holds, the spell's frames become Y's. Y stays the owner, so a following Y spell
  simply continues Y's possession.
- **Not control:** `OtherBallAction` (an unclassified touch; 159 of them are flagged
  `DefensiveClearance`), `BallDeflection`, tackles where the winner only touched the ball
  (`ballcontactSucceeded` / `layoff` without possession change, by a player without control),
  aerial duels, fouls (a free kick follows after D anyway), and `Delete` events. These are the
  challenge's "touch without control", "deflection that does not intercept" and "aerial duel
  without later control".
- **Processing:** one left-to-right pass per graded run.
- **Parameters:** 3 in total. TOL = 0.5 s (the research report's "≤ 0.5 s" and the residual
  alignment error), HOLD = 3.0 s (the report's Idea 1), and the near-ball radius of 1.5–2 m used
  for alignment.

The rules were fixed before any candidate was scored. There was one change after first results:
the `withBallControl` duel role became a control action. Before that change, only 21–36% of the
1–3 s opponent spells had any control action, and DFL's own "with ball control" flag was being
ignored. The labels without that rule are kept as `variants/v1_no_duel_role`. All conclusions
below are the same with them.

**What the rule does.** Of 1903 contested spells, the share kept rises with duration: 0–0.5 s 7%,
0.5–1 s 9%, 1–2 s 21%, 2–3 s 36%, and ≥ 3 s 100% (by C2). These figures are from the v1 rule. The
evidence behind the kept short spells is mostly passes, then won tackles, ball claims, dribbles,
shots and crosses.

## Results (final rule, 7 games x 3 degrade_v2 seeds, LOGO predictions from `probs/`)

### Possessions per 90 min

| labels | raw DFL | fold 1 s | fold 2 s | fold 3 s | **event labels** |
|---|---|---|---|---|---|
| per 90 | 412 | 344 | 272 | 224 | **262** (1943 spells) |

With other parameters: HOLD 1 s 357, 2 s 297, 4 s 245, no HOLD 231; OtherBallAction counted as
control 330; TOL 0.25 s 261, TOL 1.0 s 267; snapped events only 260.

### How well `control.fold(T)` of the DFL labels matches the event labels

| T (s) | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| frame agreement | .958 | .959 | .962 | .965 | .972 | **.979** | .978 | .964 | .952 | .926 |
| spellF1 vs event labels | .726 | .758 | .797 | .832 | .867 | **.895** | .894 | .841 | .799 | .716 |

No fold reproduces the event labels: even the best one has spellF1 0.895, so about 10% of spells
differ. Events keep some short real changes, and hand back some long ones. With HOLD 3 s the
closest T is 2.5–3. With HOLD 2 s it is T = 2.0 (spellF1 .942).

### Candidates against the event labels

| candidate | n | evt spellF1 | evt macroF1 | ctrl_mean | raw | poss / seed |
|---|---|---|---|---|---|---|
| selftrain_ctrl (only 3 game-seeds saved) | 3 | 0.7475 | 0.934 | 0.721 | 0.556 | (1939) |
| **ensemble_ctrl** | 21 | **0.7463** | 0.926 | **0.728** | 0.622 | 1924 |
| decision_tree_a (not on the list) | 21 | 0.7221 | 0.914 | 0.694 | 0.582 | 1850 |
| ensemble_vote | 21 | 0.7018 | 0.923 | 0.696 | 0.656 | 2246 |
| learned_dfl_evidence | 21 | 0.6836 | 0.914 | 0.672 | 0.616 | 2159 |
| handover_mstcn | 21 | 0.6804 | 0.919 | 0.678 | 0.652 | 2343 |
| decision_tree_b | 21 | 0.6783 | 0.900 | 0.654 | 0.544 | 1774 |
| temporal_tcn | 21 | 0.6705 | 0.916 | 0.671 | 0.645 | 2386 |

The ranking of all 18 complete prediction sets under the event labels and under ctrl_mean agrees
with Spearman 0.88 (Kendall 0.75). Full table: `arch/control_events/analysis.txt`.

Paired differences. Each pair is compared on the same game and seed. The SE is over the 7 game
means, and "wins" counts game-seeds out of 21.

| pair | evt | wins | ctrl_mean | wins |
|---|---|---|---|---|
| ensemble_ctrl − ensemble_vote | **+0.045 ± 0.005** | 21/21 | +0.032 ± 0.003 | 21/21 |
| handover_mstcn − ensemble_vote | −0.021 ± 0.008 | 2/21 | −0.019 ± 0.007 | 3/21 |
| temporal_tcn − ensemble_vote | −0.031 ± 0.006 | 1/21 | −0.025 ± 0.005 | 1/21 |
| learned_dfl_evidence − ensemble_vote | −0.018 ± 0.004 | 3/21 | −0.024 ± 0.003 | 1/21 |
| decision_tree_b − ensemble_vote | −0.024 ± 0.008 | 4/21 | −0.042 ± 0.008 | 0/21 |
| decision_tree_b − ensemble_ctrl | −0.068 ± 0.007 | 0/21 | −0.074 ± 0.007 | 0/21 |
| selftrain_ctrl − ensemble_ctrl | +0.023 (3 game-seeds only, not resolved) | 3/3 | +0.011 | 3/3 |

How robust are these rankings across label definitions (`variants/*.txt`)?

- **Stable under every variant:** ensemble_vote beats handover_mstcn, temporal_tcn and
  learned_dfl_evidence.
- **Not stable when the labels keep many spells:** ensemble_ctrl beats ensemble_vote under HOLD
  2/3/4 s and no HOLD (+0.02 to +0.05, ≥19/21), but **reverses** under HOLD 1 s (−0.026) and with
  touches counted (−0.013).
- **Most sensitive:** decision_tree_b, from −0.11 (HOLD 1 s) to +0.02 (no HOLD) against
  ensemble_vote, because it predicts the fewest spells.

The general rule is that comparisons between outputs of different granularity depend on where the
label draws the control line, and comparisons at equal granularity do not.

### Fold threshold for ensemble_vote's output (nested leave-one-game-out)

Mean spellF1 against the event labels of `control.fold(ensemble_vote, T)`:

| T | 0 | 1 | 1.5 | 2.0 | 2.25 | 2.5 | 2.75 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|---|---|
| evt | .702 | .703 | .725 | .746 | .758 | .758 | **.759** | .752 | .728 |

T is chosen on 6 games and scored on the 7th. The picks were 2.25, 2.75, 2.75, 2.75, 2.75, 2.75
and 2.25. Nested score: 0.7557 against 0.7463 for the fixed 2.0, a gain of **+0.009 ± 0.007
(4/7 games): not resolved**. For handover_mstcn and temporal_tcn the nested choice is 2.75 and
2.5, at +0.019 ± 0.005 and +0.018 ± 0.007 over 2.0.

### Robustness: which output fold works for every reading of "controlled"?

`robust_T.txt` scores ensemble_vote folded at T against nine label sets:

- ctrl1, ctrl2, ctrl3;
- the event labels with HOLD 1, 2, 3, 4 s and with no HOLD;
- the event labels with touches counted.

| output T | 1.0 | 1.25 | 1.5 | 1.75 | **2.0** | **2.25** | 2.5 | 2.75 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|
| worst-case regret | .120 | .104 | .086 | .066 | **.048** | **.047** | .062 | .077 | .089 |
| mean over the 9 label sets | .699 | .709 | .715 | .720 | .724 | **.725** | .720 | .716 | .708 |

The best output T follows the label definition almost one-to-one:

| label definition | best output T |
|---|---|
| ctrl1, HOLD 1 s, touches counted | 1.25 |
| ctrl2, HOLD 2 s | 2.0 |
| ctrl3, HOLD 3 or 4 s | 2.75 |
| no HOLD | 3.0–3.5 |

So the event data **does not remove the free parameter; it moves it.** Which spells count as
controlled is pinned down by the events, but how long an event-less spell must last (HOLD), and
whether an unclassified touch counts, still decide the answer. The challenge text leaves exactly
those two points open.

### Per-spell keep-or-hand-back rule vs the fixed fold (nested LOGO)

For every contested spell in ensemble_vote's output, a logistic regression decides whether to keep
it or hand it back. Features:

- log duration;
- the mean and max ensemble margin;
- ball visibility;
- own-minus-opponent nearest distance to the ball;
- the previous spell's duration;
- what comes next.

The target is "the event labels give these frames to the spell's team". The decision threshold is
chosen on the 6 training games.

| | nested spellF1 vs evt |
|---|---|
| logistic per-spell rule | 0.7557 |
| the same single-pass rule using duration only (T chosen on 6 games) | 0.7557 |
| `control.fold` 2.0 s | 0.7463 |

The logistic rule beats fold 2.0 by +0.009 ± 0.004 (6/7), but it equals the duration-only rule
(+0.0000 ± 0.004, 4/7). Duration dominates the coefficients (≈1.9 standardised against ≈0.6 for
the margin and ~0 for the rest). With these features, a spell-level classifier adds nothing beyond
choosing T.

## Conclusions

1. **Possessions per 90 with event-defined control: 262** (231–357 depending on the reading).
   Raw DFL is 412, fold 2 s is 272 and fold 3 s is 224. The `ctrl_mean` range of T = 1–3 s
   brackets every reasonable event reading.
2. **Selection:** keep `ctrl_mean` as the primary criterion and report the event-label score
   (`score_events.py`) next to it. Treat a candidate as better only if it does not lose
   significantly on either. The two agree on every resolved comparison among same-granularity
   models: ensemble_vote > handover_mstcn, temporal_tcn, learned_dfl_evidence. Both rank
   ensemble_ctrl first among the complete sets. Switching to event labels alone would just swap
   one guess (T) for another (HOLD) and add alignment noise.
3. **Keep the shipped fold at 2.0 s.** Against the main event labels, 2.25–2.75 s scores higher,
   but the nested gain is not resolved (+0.009 ± 0.007, 4/7). Across all nine label readings,
   2.0 and 2.25 have the smallest worst-case regret (0.048 / 0.047) and the best mean. A move to
   2.25 would be defensible but is within noise. Do not go to 2.75 or above: it loses 0.04–0.05
   if "controlled" is read loosely (HOLD ≤ 1 s or touches counted).
4. A per-spell classifier on simple features does not beat a duration threshold. Further gains
   need features that carry control evidence that duration does not, such as touch detection or
   passes completed to a teammate (research Ideas 2 and 4).
