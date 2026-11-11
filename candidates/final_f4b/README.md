# final_f4b: the previous submission ("F4b", round 5, submission v7)

> **Superseded (10 Nov 2026) by [`../final_f4sp`](../final_f4sp/README.md) (F4sP, submission v8)**:
> the same pipeline with the MS-TCN fine-tuned with SkillCorner games added and a 19.3M-parameter
> pointer, and its own decoder pick. `make_submission.py` now runs F4sP. This folder is kept
> unchanged, and `pipeline.py` here still reproduces F4b's CSV (sha256 `228fe6a4…`, below).

`pipeline.py` was what `make_submission.py` ran for v7. It applies minimum-Bayes-risk (expected-spellF1)
decoding (`../mbr_decode`) to four members, then dead ball before each half's kick-off
(`ensemble_ctrl_v6/v6_core.prekick_dead`). It is numpy + LightGBM only and deterministic, with a
fixed sampling seed. Member probabilities are rounded to float32 before decoding, so the live
decode equals the offline one on the benches' saved probabilities.

| slot | member file | what it is | weights |
|---|---|---|---|
| TCN | `adapt/member_tcn_tta4_a50.py` | `o_ls05`: the TCN recipe with label smoothing 0.05 (`nn_pff_tcn/pipeline_ls.py`), averaged over the input and 4 smoothly jittered copies, with its feature normalisation moved half-way (a = 0.5) towards the game's own label-free feature statistics (`../adapt/README.md`) | `nn_pff_tcn/tcn_ls.npz` (0.9 MB) |
| MS-TCN | `nn_pff/pipeline_c192ls.py` | `m5c_C192ls`: the MS-TCN recipe at width 192 with label smoothing 0.05 | `nn_pff/mstcn_c192ls.npz` (10.6 MB) |
| radar | `ball_radar_pff/pipeline.py` | `radar_pff`: Ball Radar-style network + stacked LightGBM | `radar.npz`, `model_br*.txt` (4.8 MB) |
| pointer | `nn_pointer/pipeline_ballc.py` | `r5_ptr_ballc`: the 8 tracked players nearest the ball, relative to the ball, plus ball kinematics; a pointer over them plus a dead class, summed per team | `nn_pointer/ptr_ballc.npz` (3.2 MB) |

Every member was trained on the 7 DFL + 48 PFF World Cup 2022 games, with DFL leave-one-game-out and
PFF game-grouped fold models resolved through `pff_folds.resolve` on the benches. None was trained
on FOOTPASS, on SkillCorner or on the 16 PFF holdout games.

Decoder config: `src hmm, alpha 1.0, rho 0.2, margin 0.0`. This is the full-DFL pick for this
member set from the g1 grid, which was fixed before any result (`../mbr_decode/README.md`). Scoring
used the nested pick per DFL game (`../nn_pointer/finals.py`).

## How it was chosen

This recipe follows the final-recipe addenda of `../../docs/prereg/round5_prereg.txt` (10 Nov
01:32 and 02:26). The final was built slot by slot under that recipe rather than chosen by its rank
in the results table, but the recipe was not written blind: F3's exact member set had already been
scored and was among the top rows 11 minutes before the 01:32 addendum, the MS-TCN slot's deciding
member-alone result was already logged, and only the MS-TCN and pointer slots had member-level
gates (the TCN slot copied the then-leader's post-hoc settings; the radar slot had no test). A
4-member MBR ensemble with the first pointer had also been scored before the 02:26 addendum. By
then 74 vote sets, 72 decoder variants, about 40 adaptation variants and 49 architecture screens
had been compared. F3, the same recipe without the pointer, was the pick until the ball-centred
pointer passed its SkillCorner gate. F4b then replaced F3 because its DFL + SkillCorner gain sum
over F3 was +0.0082 ± 0.0031 and Metrica was not worse (+0.0118). By bench, F4b vs F3 is DFL
+0.0075 ± 0.0022 (17/21), SkillCorner +0.0007 ± 0.0022 (12/20), Metrica +0.0118: the pointer's gain
is on DFL and Metrica, nil on SkillCorner, and DFL is the bench reused for design. The full
account, including the caveats, is in `../../TECHNICAL.md`, section 6.2. Results:
`../../docs/prereg/finals_report.txt`.

| paired vs v6 (`ensemble_ctrl_v6`) | F4b |
|---|---|
| DFL fine4, 21 game-seeds, decoder settings nested | 0.7555, +0.0269 ± 0.0032 (20/21) |
| SkillCorner fine3, 20 games | 0.6150, +0.0186 ± 0.0040 (17/20); at matched possession count +0.0166 ± 0.0034 |
| Metrica dev (Sample Games 1+2) | 0.7416, +0.0131 |
| FOOTPASS holdout, 13 games, run once on 10 Nov (local; data under NDA) | 0.8046 vs 0.7744, +0.0302 ± 0.0045 (13/13) |

## Checks

- **Live = offline.** Live codes equal the offline decode on DFL J03WN1 (3 seeds) and on
  SkillCorner games 1874553 and 1886347 (`live_vs_offline.py`).
- **Match** (`match_run.py`: positions and tags only, no labels):
  - 322 possessions (273.6 per 90), dead share 0.455;
  - 3.47% of graded frames differ from v6;
  - CSV sha256 `228fe6a47e29a411746a97e9c7f758ad557c2f6acf768343b39932ede8ac4f30`.
- **Byte-identical CSV:**
  - macOS arm64 (Accelerate): `make_submission.py` (always one thread), and `match_run.py` (the
    same pipeline without the thread pin) twice and with `OMP_NUM_THREADS` = 2 and 8;
  - Linux arm64 (OpenBLAS) in a container;
  - Linux x86-64 under Rosetta emulation (OpenBLAS) in a container;
  - a fresh copy of the repository in a Linux arm64 container with only
    `requirements-submission.txt`.

  Native x86-64 hardware was not tested.
- **Noise probe.** Multiplying the member probabilities by (1 + 1e-6 u), u ~ U(−1, 1), over 3
  draws changes 0 frames.
- **Run time:** about 4-7 minutes, single thread.

## Files

| file | role |
|---|---|
| `pipeline.py` | the submission pipeline (bench interface: `predict`, `predict_proba`) |
| `live_vs_offline.py` | live codes vs the offline decode of saved member probabilities |
| `match_run.py` | label-free BAR-ATH run: timing, statistics, sha256, noise and rounding probes |

Needed beside this folder: `../mbr_decode/`, `../adapt/`, `../nn_pointer/`,
`../nn_pff/pipeline_c192ls.py` + `mstcn_c192ls.npz`, `../nn_pff/pipeline_w.py`,
`../nn_pff_tcn/pipeline_ls.py` + `tcn_ls.npz`, and the folders v6 already uses (`ensemble_ctrl_v6`,
`nn_pff`, `nn_pff_tcn`, `ball_radar_pff`, `learned_dfl_evidence`).
