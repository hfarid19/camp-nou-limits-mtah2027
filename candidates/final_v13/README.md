# final_v13: submission v13 (v12 with unequal member weights in the decoder)

`pipeline.py` is v12 (`../final_v12`) with one change: the five members no longer have equal weight in the
minimum-Bayes-risk decode. Nothing is retrained. The four members, the stacker file (`../final_v12/stacker_v12.txt`),
the decoder setting (`src hmm, alpha 0.1, rho 0.2, margin 0.0`, fixed sampling seed) and dead ball before each
half's kick-off are v12's.

| member | TCN | MS-TCN | radar | pointer | stacker |
|---|---|---|---|---|---|
| weight | 1 | 3 | 1 | 2 | 3 |

A weight w means the member is entered w times in the decoder's list (its probabilities and its codes), as in
the member-weight search below. The list is built in "rounds": the members with weight ≥ 1, then those with
weight ≥ 2, then those with weight ≥ 3, each round in v12's member order. So the decoder reads 10 entries:

    TCN, MS-TCN, radar, pointer, stacker,  MS-TCN, pointer, stacker,  MS-TCN, stacker

The stacker still reads the four members once each, so its inputs do not change. `predict_proba` gives the
unweighted vote shares of the five members' codes; `make_submission.py` does not use it.

**This weighting was chosen after seeing results, and it failed the rule written before the search.** Rule
file: `../../docs/prereg/round5_prereg.txt`, addenda of 14 Nov 2026 12:20 and 15:53 CST, the RESULT that follows
them, and the DECISION of 16:20 CST.

**Status: the submission since 14 Nov 2026 (the user's choice).** v12 was published on 14 Nov and replaced the
same day. `make_submission.py` runs v13 by default (`DEFAULT_CANDIDATE = "final_v13"`).

```sh
python make_submission.py                                           # writes evaluation/results/camp_nou_limits.csv
python make_submission.py --candidate final_v13 --out <path>.csv   # the same pipeline, to another path
python make_submission.py --candidate final_v12 --out v12.csv      # v12, equal weights
```

## How the weighting was found

On 14 Nov we searched member weights for v12, decode-only, on the saved v12 benchmark outputs of the five
members. Every weight from 0 to 3 for each member gives 4^5 − 1 = 1,023 weightings. Each was compared with equal
weights on four benchmarks, which is 4,088 benchmark-level comparisons. These benchmarks had already been used for
every earlier choice. Equal weights reproduced v12's scores exactly (0.8197 / 0.7735 / 0.6333 / 0.8413), and
"stacker × 2" reproduced the earlier stacker-twice result, on all 95 units. Work files:
`$MTAH_WORK/arch/r5_arch/weights_v12/out/analysis_max3.md` and `grid_max3.csv`.

**The rule, written at 12:20 before any run.** The search is descriptive. A weighting is "worth proposing" only
if its *nested* gain over equal weights is above 2 standard errors on at least two benchmarks, including Metrica
or SkillCorner, and no benchmark is below −1 standard error. The selection criterion was fixed as the mean z
(gain divided by its standard error) across the four benchmarks.

**The pick and the nested result.** On all four benchmarks the mean-z criterion picks 1 3 1 2 3. It comes top
mainly because its Metrica gain has a very small standard error (+0.0113 ± 0.0004 over 3 games, z +28). To
estimate the gain honestly, the pick was made on three benchmarks and scored on the fourth (leave one benchmark
out):

| held-out benchmark | weighting picked on the other three | gain on the held-out benchmark ± SE (wins) | z |
|---|---|---|---|
| Metrica (3) | 2 2 0 3 3 | +0.0055 ± 0.0029 (3/3) | +1.93 |
| DFL (21 units, 7 games) | 1 3 1 2 3 | +0.0004 ± 0.0009 (13/21) | +0.46 |
| SkillCorner (20) | 1 3 1 2 3 | +0.0013 ± 0.0018 (12/20) | +0.75 |
| PFF (48) | 1 3 1 2 3 | −0.0022 ± 0.0007 (20/48) | −3.21 |

No benchmark is above 2 standard errors and PFF is below −1, so **the rule is not met**. Leave one game out,
picking on all other games of all four benchmarks, gives Metrica +0.92 and the same values elsewhere; it also
fails. Picking on the other games of the same benchmark only does meet the rule, but each benchmark then picks a
different weighting (SkillCorner 0 3 3 1 1, PFF mostly 3 1 1 3 1), and each costs about 0.01 on the others, so it
does not give one weighting for the match. The 15:53 trigger for a finer second pass (nested gain above +1 SE on
two benchmarks, none below −1) also failed, and that pass was not run. The analysis concluded that no weighting
is worth proposing and that v12 keeps equal weights.

**The decision.** At 16:20 CST on 14 Nov, after seeing these results, the team chose 1 3 1 2 3 anyway. This is a
choice made after results, against our own rule, on benchmarks reused for every choice. The rule file records it
before the candidate was built.

**Other findings of the search.**
- The gain is almost only on Metrica, which is three games.
- The MS-TCN is fine-tuned on Metrica and the stacker is trained on Metrica. Weighting these two up may simply
  favour Metrica. This is a hypothesis; we did not test it.
- The benchmarks prefer opposite weightings. SkillCorner prefers more radar and less pointer. The World Cup games
  (PFF) and the Bundesliga (DFL) prefer more pointer and more TCN.
- Equal weights rank 286 of 1,023 by the mean of the four benchmarks. The best in-sample mean is 0 1 0 2 3
  (Metrica +0.0201 ± 0.0063, the other three within ±0.002).
- Deviations recorded in the rule file: scale multiples (for example 2 2 2 2 2 and 1 1 1 1 1) were not removed,
  because they differ slightly through the vote's tie-break; the 0..2 grid was run and its in-sample table seen
  before the nested result of the full grid.

## Evidence

The benchmarks and units are v12's: Metrica strict leave-one-game-out (3 games), DFL fine4 (21 game-seeds, SE over
7 games), SkillCorner fine3 (20 games), PFF fine3 (48 games). **v13's scores are in-sample for the weight choice.**

| bench | v12 | **v13** | v13 − v12 ± SE (wins) | macroF1, v13 − v12 |
|---|---|---|---|---|
| Metrica strict, 3 games | 0.8197 | **0.8310** | +0.0113 ± 0.0004 (3/3) | +0.0064 |
| DFL fine4, 21 game-seeds | 0.7735 | **0.7739** | +0.0004 ± 0.0009 (13/21) | −0.0003 |
| SkillCorner fine3, 20 games | 0.6333 | **0.6346** | +0.0013 ± 0.0018 (12/20) | +0.0009 |
| PFF fine3, 48 games | 0.8413 | **0.8391** | −0.0022 ± 0.0007 (20/48) | +0.0001 |

v13 gains on Metrica, is within noise on DFL and SkillCorner, and loses on PFF by about 3 standard errors.

v13 was scored once on the public 5 minutes, on 14 Nov 2026, after it had been chosen and checked (announced in the rule file beforehand): spellF1 0.6595, macroF1 0.8332, 20 predicted possessions against 17 true (v12: 0.6256 / 0.8333 / 22). The window was not used to choose it and the score decided nothing.

## The match (label-free)

309 possessions (262.6 per 90 graded minutes), mean length 11.1 s, dead share 0.460. v12: 319 possessions
(271.1 per 90), dead share 0.455. v13 has 131 stoppages against v12's 136. None of them is new: 133 of v12's 136
stoppages are kept, some of them merged. v13 differs from v12 on 1.67% of graded frames (2,650 frames: 1,347 turn
to dead ball, 500 turn from dead ball to a team, and 803 change team).

CSV sha256 `697fae9f6c9dfb832fadf0ead231be1db3592b1b078b2e56fd5a28ece4215487`. One run on macOS arm64 took 246 s
with a peak of 6.0 GB.

## Final checks

Every run below was made on 14 Nov and wrote exactly the same file (sha256 `697fae9f…215487`), and the file passes `evaluation/validate_submission.py`.

| Check | Platform | Run time | Peak memory of the main process |
|---|---|---|---|
| `make_submission.py`, twice in a row | macOS arm64 (Accelerate), 1 thread | 4.1 and 4.1 minutes | 6.0 and 6.0 GB |
| The same steps with 1, 2 and 8 threads, through a driver outside the repository | macOS arm64 | 4.1, 3.9 and 3.9 minutes | 5.8, 5.2 and 5.9 GB |
| Fresh clone in a new environment | macOS arm64 | 4.4 minutes | 6.0 GB |
| The same clone in a `python:3.14.6-slim` container with `libgomp1` | Linux arm64 (OpenBLAS) | 10.3 minutes | 3.4 GB |
| The same clone in an x86-64 container under Rosetta emulation | Linux x86-64 | 18.8 minutes | 3.3 GB |
| The numpy pointer backend (`MTAH_PTR_BACKEND=numpy`) | macOS arm64 | 20.1 minutes | 6.8 GB |

The fresh clone was the previous commit (`59b26af`) plus v13's two code files, `make_submission.py` and `candidates/final_v13/pipeline.py`, copied in before v13 was committed. Live and offline decoding gave identical labels on the Bundesliga game J03WN1 for 3 seeds, and the five members' live outputs were bitwise equal to v12's. A second, independent decode of v12's saved member outputs with the weights 1 3 1 2 3 also wrote the same file. Not checked: native x86-64 hardware, live against offline on Metrica or SkillCorner.

## Training data

As v12: nothing is retrained. No model has seen the match, FOOTPASS or the 16 PFF holdout games.

## Files

| file | role |
|---|---|
| `pipeline.py` | the v13 pipeline (bench interface: `predict`, `predict_proba`); `WEIGHTS = (1, 3, 1, 2, 3)` |
| `../final_v12/stacker_v12.txt`, `stacker_v12_meta.json` | the stacker, unchanged (its sha256 is checked against the meta file) |

Nothing that v8, v9, v10, v11 or v12 runs was changed.
