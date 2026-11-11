# mbr_decode: expected-spellF1 (minimum-Bayes-risk) decoding of the ensemble (round 5, track R5-1)

The decoder works on the members' saved per-frame probabilities and decoded codes. It is numpy only and
deterministic, with a fixed seed.

> **Status (10 Nov 2026):** this decoder is part of the submission, `../final_f4sp` (F4sP, v8), on a
> different member set and with that set's own full-DFL pick from grid g1 (alpha 0.1, rho 0.2,
> **margin 0.0**; the previous submission `../final_f4b`, F4b, picked alpha 1.0, rho 0.2, margin 0.0).
> "Shipped" below means this track's pick for the lsw members (`pipeline_mbr.py`), which was not the
> final submission.

## Method (`mbr.py`, `decoder.py`)

1. **Candidates.** These are complete label sequences for the game:
   - the hard vote of the members under the v6 rule (merge_short_v2 1 s, then control.fold 1.25 s). This is
     the **base**.
   - the same vote under the 11 other rules of the (merge {0, 1 s}) x (fold {0, 0.5, 0.75, 1, 1.25, 1.5 s})
     grid;
   - each member alone, raw and under the v6 rule.

   Duplicate candidates are dropped.
2. **Posterior samples.** We draw 32 forward-filtering backward-sampling paths from a 3-label HMM. Its
   emissions are the members' mean probabilities raised to the power `alpha` (tempering), summed over
   5-frame steps. Each label has 4 sub-states, which gives a negative-binomial duration with exit
   probability `rho` per step. The seed is fixed (2027). Every sample is then read at the three label
   granularities the benches use: control.fold at T = 0, 0.5 and 1 s.

   With `src=mem`, the members' own codes are the samples instead; `src=mix` uses both sources.
3. **Utility.** Mean spellF1 over the IoU thresholds equals 2·Σ(matched fraction per spell) / (n_pred +
   n_gt). For each candidate spell, `spell_util` computes the fraction of the thresholds 0.3 … 0.7 at
   which it is matched to a sample spell (mutual best IoU, same team). At IoU ≥ 0.5 this is exact. It
   reproduces `possession_metrics.spell_scores` exactly on the bench targets.
4. **Window selection.** A window is a stretch between frames where every candidate says D, so no
   candidate spell crosses a window edge. In each window we keep the candidate with the highest
   `2·U − F·n`:
   - U = expected matched spells;
   - n = number of spells;
   - F = the current expected spellF1, iterated (Dinkelbach linearisation of the F1 ratio).

   A non-base candidate must beat the base by more than `margin`.
5. **prekick_dead**, as in v6 (live only; it does nothing offline).

## Protocol and multiplicity

- The hyper-parameters are chosen NESTED on DFL: for each DFL game, the config with the best mean fine4 on
  the other 6 games. SkillCorner and Metrica were never used to choose them.
- The full-DFL pick is shipped and applied unchanged to SkillCorner, Metrica and PFF.
- The references are the offline votes with the v6 rule. They reproduce `swap_eval.py`'s numbers:
  - v6: DFL 0.7286, SkillCorner 0.5964, Metrica 0.7285;
  - lsw: 0.7385, 0.6018, 0.7320.

| grid | configs | status |
|---|---|---|
| **g1** | 21 = alpha {0.1, 0.3, 1} x rho {0.05, 0.2} x margin {0, 0.25, 0.5} (hmm), plus mem x 3 margins | fixed before any result (only a one-unit smoke test) |
| g2 | 14 = alpha {0.3, 1} x rho {0.2, 0.5} x r {2, 4} x margin {0, 0.5} (new combinations), plus mix x 2 | exploratory, written after seeing g1 |
| g3 | 2 = single fold read (T = 0 or 1 s) at alpha 1, rho 0.2, margin 0 | exploratory, written after the coordinator's "fewer knobs" question |
| stab1 | seeds 1-5 and N = 64 / 128 of the shipped config | stability only, not selection |

- g1 and g2 were each run on both member sets, v6 and lsw.
- In total, 72 decoder variants were scored (g1 42, g2 28, g3 2), plus the stability runs.

**Coordinator ruling.** g1 is the track's grid because it was fixed before any result. g2 is
exploratory: it is reported, but it can neither make nor break eligibility.

**This ruling was made after the SkillCorner numbers of g1 and g2 were known.** The pooled
g1+g2 full-DFL pick (alpha 0.3, rho 0.5, r 4, margin 0.5) has these results, paired vs v6:
- DFL: nested +0.0146 ± 0.0037;
- SkillCorner: −0.0070 ± 0.0022, which fails the prereg.

So a DFL-nested choice can land on a config that hurts SkillCorner.

## Results (lsw members, g1, nested; paired, SE over games)

| | DFL fine4 (21 game-seeds) | SkillCorner fine3 (20) | Metrica dev (4) | PFF OOF fine3 (48, screen) |
|---|---|---|---|---|
| v6 (offline vote) | 0.7286 | 0.5964 | 0.7285 | 0.7815 |
| lsw (offline vote) | 0.7385, +0.0099 ± 0.0021 (17/21) | 0.6018, +0.0054 ± 0.0026 | 0.7320, +0.0035 | 0.7891, +0.0076 ± 0.0016 |
| **MBR-lsw vs v6** | **0.7423, +0.0137 ± 0.0044 (16/21)** | **0.6087, +0.0123 ± 0.0033 (16/20)** | **0.7376, +0.0091 ± 0.0097** | 0.8030, +0.0215 ± 0.0018 (46/48) |
| MBR-lsw vs lsw | +0.0039 ± 0.0043 (13/21) | +0.0069 ± 0.0023 (13/20) | +0.0056 ± 0.0054 | +0.0139 ± 0.0014 (45/48) |
| MBR on v6 members vs v6 | +0.0078 ± 0.0059 (15/21) | +0.0066 ± 0.0025 | −0.0005 | +0.0125 ± 0.0018 |

**Nested picks on DFL (lsw members).** All 7 folds chose rho 0.2 (hmm), as follows:

| config | folds |
|---|---|
| alpha 0.3, margin 0.25 | 2 |
| alpha 1.0, margin 0 | 2 |
| alpha 1.0, margin 0.5 | 1 |
| alpha 0.1, margin 0.25 | 1 |
| alpha 0.3, margin 0 | 1 |

**Full-DFL pick (shipped):** alpha 1.0, rho 0.2, margin 0.5 (r 4, N 32, stride 5, folds 0/0.5/1, seed 2027).

**Per-config table over g1** (`$MTAH_WORK/arch/mbr_decode/g1_table.txt`). Number of the 21 configs that
are positive:

| | SkillCorner | DFL | Metrica |
|---|---|---|---|
| lsw members, vs v6 | 21 | 18 | 17 |
| lsw members, vs lsw | 18 | 15 | 14 |

- Every hmm config with rho 0.2 is positive on all three benches vs v6.
- rho 0.05 at alpha ≤ 0.1 hurts DFL and Metrica but is the best on SkillCorner (+0.023). SkillCorner
  rewards smoother output.
- `mem` (no HMM) gives +0.0145 on DFL but 0.000 on SkillCorner.

**Coarseness and matched count** (`analysis.py`, possessions per 90 of graded time):

| | DFL | SkillCorner | Metrica |
|---|---|---|---|
| v6 | 308.2 | 291.7 | 285.8 |
| lsw | 305.4 | 284.1 | 280.7 |
| MBR | 312.7 | 288.2 | 288.4 |
| MBR vs v6, at matched count (v6 re-folded per unit to MBR's count) | +0.0129 ± 0.0037 | **+0.0100 ± 0.0024 (16/20)** | +0.0070 |
| MBR vs lsw, at matched count | +0.0011 ± 0.0028 | +0.0077 ± 0.0023 | +0.0003 |

- MBR is not coarser than v6 on SkillCorner.
- On DFL, re-folding lsw to MBR's count per game recovers most of MBR's gain over lsw. Note that this
  re-fold uses MBR's count.

**Label readings on DFL** (MBR, paired):

| reading | vs v6 | vs lsw |
|---|---|---|
| raw | +0.0157 | +0.0074 |
| T0.5 | +0.0170 | +0.0070 |
| T1 | +0.0103 | −0.0008 |
| evt1 | +0.0119 | +0.0018 |
| T1.5 | +0.0003 | **−0.0089 ± 0.0036** |
| T2 | +0.0030 | −0.0025 |
| evt3 | +0.0056 | +0.0018 |
| evt2 | +0.0077 | +0.0007 |
| sel2 | +0.0054 ± 0.0044 | −0.0019 ± 0.0032 |

- Against v6, every reading improves.
- Against lsw, the gain sits on the fine readings, with a loss at T1.5. This is a mild shift towards
  finer granularity.

**Fewer knobs.**
- In g1, alpha 1 (no tempering) with margin 0 leaves rho as the only knob:
  - rho 0.2: DFL +0.0163, SkillCorner +0.0113;
  - rho 0.05: DFL +0.0152, SkillCorner +0.0147, Metrica +0.005.

  Both are close to the pick.
- g3 (exploratory), a single fold read at alpha 1, rho 0.2, margin 0:
  - T 0: DFL +0.0168, SkillCorner +0.0101;
  - T 1: DFL +0.0148, SkillCorner +0.0146, Metrica +0.0065.

**Seed stability** (shipped config, seeds 2027 and 1-5):

| bench | gain vs v6 | SD over seeds | gain vs lsw | SD over seeds |
|---|---|---|---|---|
| DFL | +0.0159 | 0.0003 | +0.0060 | 0.0003 |
| SkillCorner | +0.0121 | 0.0004 | +0.0067 | 0.0004 |

N = 64 or 128 changes nothing. The seed is fixed, so the output is a deterministic function of the
inputs.

**Prereg** (`../../docs/prereg/round5_prereg.txt`).
- Route (a) holds:
  - DFL +0.0137 = 3.1 SE;
  - SkillCorner +0.0123 (not worse);
  - Metrica +0.009.
- Route (b) also holds:
  - SkillCorner +0.0123 = 3.7 SE;
  - at matched count +0.0100 ± 0.0024.
- The final-pick rule is the DFL + SkillCorner gain sum vs lsw: +0.0039 + 0.0069 = +0.0108. The SE of
  the difference is about 0.0049, so the gain is about 2.2 SE.

## Live pipeline: `pipeline_mbr.py`

- Members: lsw (`nn_pff_tcn/pipeline_ls.py`, `nn_pff/pipeline_w.py`, `ball_radar_pff/pipeline.py`).
- Each member runs `predict` (codes) and `predict_proba`. Probabilities are rounded to float32, as the
  benches saved them. Then `decoder.decode(CFG)` runs, followed by `prekick_dead`.
- **Live = offline:** `bench_skc.py run` on SkillCorner games 1874553 and 2016236 gives codes identical to
  the offline decode from the saved probabilities.
- **BAR-ATH** (`match_run.py`: positions and tags only, no labels):
  - MBR: 334 possessions (283.8/90), dead share 0.465;
  - lsw: 331 (281.3/90), dead share 0.466;
  - 1.01% of graded frames changed (27 of 120 windows switched);
  - timing: members 45 s, MBR decode 0.7 s (199,319 frames);
  - a repeat decode is identical.
- Not wired into `make_submission.py` itself; the submission (`../final_f4sp/pipeline.py`, before it
  `../final_f4b/pipeline.py`) reuses `MBRVote` with its own members and config.
- The sampler makes discrete choices on exp/log/cumsum results, so byte identity across machines
  had to be checked. That was done for both submissions that use this decoder: see the checks in
  `../final_f4b/README.md` (macOS at 1/2/8 threads, Linux arm64, Linux x86-64 under emulation, a
  fresh clone, and a 1e-6 noise probe) and `../final_f4sp/README.md`. `pipeline_mbr.py` itself (the lsw members) was not re-checked
  that way.

## Files

| file | role |
|---|---|
| `mbr.py` | spells, utility, FFBS sampler, window selection |
| `decoder.py` | candidates, samples, decode |
| `pipeline_mbr.py` | the bench / submission interface |
| `data.py` | bench units: probabilities, codes, targets, scoring |
| `exp.py` | grids g1 / g2 / g3 / stab1 on a bench |
| `report.py` | tables, nested pick |
| `analysis.py` | coarseness, matched count, readings |
| `skc_probs.py` | members' `predict_proba` on the 20 SkillCorner games → `$MTAH_WORK/probs/skc_mbr/` (the codes are copied from `probs/skc/`) |
| `match_run.py` | label-free match statistics |

Logs and results are in `$MTAH_WORK/arch/mbr_decode/`:
- `res_<grid>_<bench>_<members>.pkl`;
- `g1_table.txt`, `stab1_table.txt`, `g3_table.txt`;
- `analysis.log`, `match_run.log`.
