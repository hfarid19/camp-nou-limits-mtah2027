# final_f4sp: the submission ("F4sP", round 5, submission v8)

`pipeline.py` is what `make_submission.py` runs. It is `../final_f4b` (F4b) with two member slots
swapped and the decoder setting re-picked for the new member set: minimum-Bayes-risk
(expected-spellF1) decoding (`../mbr_decode`) of four members, then dead ball before each half's
kick-off (`ensemble_ctrl_v6/v6_core.prekick_dead`). It runs on numpy and LightGBM, plus PyTorch on
the CPU for the pointer's forward pass (float64; numpy is the fallback and the reference, see
"Inference backends" below), and is deterministic, with a fixed sampling seed. Member probabilities
are rounded to float32 before decoding, so the live decode equals the offline one on the benches'
saved probabilities.

| slot | member file | what it is | weights |
|---|---|---|---|
| TCN | `adapt/member_tcn_tta4_a50.py` | `o_ls05`, unchanged from F4b: the TCN recipe with label smoothing 0.05 (`nn_pff_tcn/pipeline_ls.py`), averaged over the input and 4 smoothly jittered copies, with its feature normalisation moved half-way (a = 0.5) towards the game's own label-free feature statistics (`../adapt/README.md`) | `nn_pff_tcn/tcn_ls.npz` (0.9 MB) |
| MS-TCN | `nn_pff/pipeline_c192ls_skcft.py` | `m5c_C192ls_skcft` (new): F4b's MS-TCN (`m5c_C192ls`, width 192, label smoothing 0.05) fine-tuned on the 7 DFL + 48 PFF games plus the 20 SkillCorner A-League open-data games: lr 5e-5 (0.1 x the parent's), 5 epochs (750 iterations at batch 8), the parent's feature normalisation kept. Same network, features and decoder as the parent | `nn_pff/mstcn_c192ls_skcft.npz` (10.6 MB) |
| radar | `ball_radar_pff/pipeline.py` | `radar_pff`, unchanged: Ball Radar-style network + stacked LightGBM | `radar.npz`, `model_br*.txt` (4.8 MB) |
| pointer | `nn_pointer/pipeline_ballc_p20m.py` | `r5_ptr_ballc_p20M` (new): F4b's ball-centred pointer (the 8 tracked players nearest the ball, relative to the ball, plus ball kinematics; a pointer over them plus a dead class, summed per team) widened 5x (D 320, heads 20, C 640, Dp 160): 19,258,736 parameters instead of 796,144. Same recipe, data and seed; no code change (every width is read from the model's config) | `nn_pointer/ptr_ballc_p20m.npz` (77.1 MB) |

The TCN, radar and pointer were trained on the 7 DFL + 48 PFF World Cup 2022 games; the MS-TCN also
on the 20 SkillCorner A-League games (SkillCorner open data, MIT licence, © 2020 SkillCorner; notice
in `../../TECHNICAL.md`, section 4). DFL leave-one-game-out and PFF game-grouped fold models are
resolved through `pff_folds.resolve` on the benches: the bench's `--folds` JSON maps each shipped
weights file's sha256 prefix to its fold folder (`f207e06efe95` → the MS-TCN's 7 DFL fold models,
`448bd43e7b6e` → the pointer's), and each fold model was checked to exclude its DFL game. No member
was trained on FOOTPASS or on the 16 PFF holdout games.

Decoder config: `src hmm, alpha 0.1, rho 0.2, margin 0.0` (F4b: alpha 1.0). This is the full-DFL
pick for this member set from the g1 grid, which was fixed before any result
(`../mbr_decode/README.md`); the nested pick was the same in all 7 DFL folds. Scoring used the
nested pick per DFL game (`../nn_pointer/finals.py`).

## Inference backends

The pointer's network forward pass (`../nn_pointer/pipeline.py`) can run on two backends; feature
building, the stride-5 phases, the team-swap average, the member's decoder and everything after it are
the same code for both.

- **torch** (the default when PyTorch is installed): `../nn_pointer/ptr_torch.py`, the maths of
  `ptr_np.forward` op for op in PyTorch on the CPU, float64. Convolutions are written as the same
  shifted matrix products as in numpy (no torch convolution kernels); oneDNN is disabled,
  `torch.use_deterministic_algorithms(True)` is on, and it uses `MTAH_TORCH_THREADS`, else
  `OMP_NUM_THREADS`, else 1 thread (`make_submission.py` pins 1). It runs in a separate worker
  process that loads only numpy and torch: on macOS, torch and LightGBM in one process hang or crash
  (their OpenMP runtimes clash; tested with torch 2.15.0 and LightGBM 4.7.0).
- **numpy**: `../nn_pointer/ptr_np.py`, float64, unchanged from submission v8 as first built. Used
  when torch is not installed, or with `MTAH_PTR_BACKEND=numpy`.

`MTAH_PTR_BACKEND` = `numpy`, `torch`, `torch32` (float32, experiments only) or `auto` (the default:
torch if it can be imported, else numpy) overrides the choice; the pipeline prints which backend ran
on stderr. Both backends give the same CSV (Checks, below). float32 was measured and not shipped:
its probabilities differ from numpy's by up to 1.3e-6 on the match (412,395 of the float32-rounded
values differ; 1.6e-6 on DFL J03WN1, 1.3e-6 on Metrica Sample Game 1), no member code changed and
the macOS CSV was the same, but it was run on macOS only, float32 matrix-product kernels differ
between CPUs and BLAS libraries, its deviations are larger than the 1e-6 the noise probe covers, and
float64 is already fast (pointer 83 s in float64 vs 41 s in float32, same loaded Mac).

The weights are read through `../chunked_weights.py`, which also accepts a file stored as
`<name>.npz.part000`, `.part001`, ... plus `<name>.npz.sha256` (for files over GitHub's 100 MB limit;
`python candidates/chunked_weights.py split <file>`). When the single file exists it is used as it
is; otherwise the parts are joined in memory and checked against the sha256. The fold-model lookup of
the benches (`dfl_folds`, `pff_folds`) keys on the sha256 of the whole content in both cases. No
shipped weights file is split at present (the largest is 77.1 MB).

## How it was chosen

F4b was the standing pick (`../final_f4b/README.md`). Two later experiments, each requested by the
user and each with a decision rule written before its first result
(`../../docs/prereg/round5_prereg.txt`, addenda of 10 Nov 11:20, 11:30 and 11:50, and 13:30 and
14:55), produced F4sP in two steps. The account in `../../TECHNICAL.md`, section 6.2, has the full
detail.

1. **"+SKC": SkillCorner as training data** (addenda 11:20 / 11:50). The two best members were
   retrained from scratch with the 20 SkillCorner games added, and separately fine-tuned from the
   finished weights (SKC-FT). SkillCorner was scored cross-validated (4 folds of 5 games; each game
   only by a model that never saw it). Member gate, alone and paired vs the parent: SkillCorner
   ≥ +0.010 and ≥ 2 SE, DFL not worse by > 1 SE, Metrica not worse by > 0.010. Only the MS-TCN
   fine-tune passed (SkillCorner-CV +0.0388 ± 0.0026, 20/20; DFL +0.0054 ± 0.0019, 13/21; Metrica
   −0.0028). The pointer fine-tune (DFL −0.0099 ± 0.0030) and both from-scratch retrains (DFL
   −0.0087 ± 0.0053 and −0.0154 ± 0.0054) failed on DFL; a parallel Metrica fine-tune experiment
   (addendum 11:50 B) had no qualifier. The one final allowed, **F4s** = F4b with the MS-TCN
   swapped, scored DFL 0.7564, SkillCorner 0.6239, Metrica 0.7397; vs F4b DFL +0.0009 ± 0.0022
   (11/21), SkillCorner +0.0089 ± 0.0018 (16/20), gain sum +0.0098 ± 0.0028 (> 2 SE), Metrica
   −0.0019, so it replaced F4b as the standing pick under the rule. F4s was never shipped.
2. **Pointer size ladder** (addenda 13:30 / 14:55). The ball-centred pointer was retrained at about
   2.4M, 5.9M, 19.3M and 58.8M parameters (widths fixed in advance, one recipe per size, no tuning).
   PFF out-of-fold screen, 48 games, paired vs the 0.8M pointer (0.8137): 2.4M +0.0050 ± 0.0015,
   5.9M +0.0100 ± 0.0018, 19.3M +0.0133 ± 0.0018, 58.8M +0.0111 ± 0.0020 (below 19.3M by
   0.0022 ± 0.0014, so the ladder stopped; the ~180M size was never trained). The 19.3M pointer then
   passed its member gate alone vs the 0.8M pointer: DFL +0.0084 ± 0.0039 (16/21; 2.15 SE against a
   2 SE bar), SkillCorner −0.0019 ± 0.0037 (10/20), Metrica −0.0074. The one final, **F4sP** = F4s
   with the pointer swapped, beat F4s on the pre-registered rule (gain sum > 2 SE, Metrica not worse
   by > 0.010). The addendum required the weights size (77.1 MB) and the CPU run time (about 24
   minutes per match for the pointer alone) to be reported before any submission change; they were,
   and the user approved building the submission on F4sP.

| bench | F4sP | vs F4s | vs F4b | vs v6 |
|---|---|---|---|---|
| DFL fine4, 21 game-seeds, decoder settings nested | 0.7615 | +0.0051 ± 0.0019 (15/21) | +0.0060 ± 0.0020 (17/21) | +0.0330 ± 0.0038 (21/21) |
| SkillCorner fine3, 20 games (MS-TCN cross-validated, see below) | 0.6285 | +0.0046 ± 0.0012 (16/20) | +0.0135 ± 0.0019 (17/20) | +0.0321 ± 0.0039 (20/20); at matched possession count +0.0298 ± 0.0032 (20/20) |
| DFL + SkillCorner gain sum | | +0.0097 ± 0.0023 | | |
| Metrica dev (Sample Games 1 + 2) | 0.7462 | +0.0064 | +0.0046 | +0.0177 |
| possessions / 90 (DFL, SkillCorner, Metrica) | 310, 285, 288 | F4s 308, 284, 286 | F4b 314, 289, 290 | |

Results: `../../docs/prereg/finals_F4sP_report.txt` and `finals_F4s_report.txt`.

**Caveats, stated before the result was used and still true:**
- **SkillCorner is no longer an untouched bench for the MS-TCN.** Its SkillCorner numbers come from
  four per-fold models (each game scored by the model that did not train on it), not from the
  shipped model, which trained on all 20 games. They are not like-for-like with SkillCorner numbers
  before the +SKC experiment, where every member was one full model that never saw SkillCorner.
- **The new pointer only just cleared its DFL gate** (2.15 SE against a 2 SE bar), and alone it was
  slightly worse on SkillCorner (−0.0019 ± 0.0037) and Metrica (−0.0074). Its gain shows up inside
  the ensemble (F4sP vs F4s: DFL +0.0051, SkillCorner +0.0046, Metrica +0.0064), on benches that were
  also used to design it.
- **No clean confirmation.** The FOOTPASS holdout confirmation was spent on F4b and was not run for
  F4s or F4sP. Every number above is from benches reused for design and selection.
- **Bench probabilities of the 19.3M pointer came from a float32 GPU path** (an experiment-only
  torch port, `../nn_pointer/pipeline_cfg_gpu.py`), not from the float64 CPU paths that ship (numpy
  or torch). Its codes were checked identical to the numpy path on sample games only (SkillCorner
  2006229 and DFL J03WR9 seed 2027: 0 frames differ, max |ΔP| 2.6e-6). The live = offline check
  below runs the numpy path against those saved probabilities; the torch path was then checked
  against the numpy path.
- **Multiplicity.** On top of round 5's earlier comparisons (`../final_f4b/README.md`), these two
  experiments add 4 +SKC member variants, 2 Metrica fine-tunes, 4 pointer sizes and 2 finals (F4s,
  F4sP).

## Checks

All on the Mac (Apple M5 Max) or in local containers; positions and tags of the match only, no labels.
The checks up to "Run time" were made when v8 was built, with the numpy backend; "Torch backend"
below repeats the ones that matter with the torch backend.

- **Live = offline.** The live codes of `pipeline.py`, run by the bench (`dump_members.py`, which
  wraps it unchanged), equal the offline decode of the members' saved bench probabilities
  (`live_vs_offline.py`) on DFL J03WN1 (all 3 degradation seeds; the bench resolved `mstcn_c192ls_skcft.npz` and
  `ptr_ballc_p20m.npz` to their leave-one-game-out `fold_tcn_1` models), Metrica Sample Game 1 (both
  seeds) and Sample Game 2 (seed 2027). Member by member, the TCN, MS-TCN and radar
  probabilities are identical to the saved ones (max |ΔP| = 0; the MS-TCN's also confirms that the
  DFL bench loaded its leave-one-game-out fold model), the 19.3M pointer's numpy float64
  probabilities differ from the saved float32-GPU ones by at most 2.9e-6, and no member's codes
  differ on any frame. SkillCorner was not used for this check: the MS-TCN's saved SkillCorner
  probabilities come from cross-validated fold models, while the pipeline loads the full model, so
  they are expected to differ.
- **Match** (`match_run.py`: positions and tags only, no labels):
  - 326 possessions (277.0 per 90), dead share 0.457 (F4b: 322, 273.6 per 90, 0.455);
  - 3.48% of graded frames differ from v6, 2.18% from F4b, 1.05% from F4s (F4s rebuilt from this
    run's first three members and F4b's saved match outputs of the 0.8M pointer);
  - CSV sha256 `60fc793e90f1df6ce5dda4b2140988fdfd1c1cfc4ee01950b10c429186dd6c57`;
    `validate_submission.py`: valid.
- **Byte-identical CSV** (sha256 above):
  - macOS arm64 (Accelerate): `make_submission.py` (always one thread), and `match_run.py` (the
    same pipeline without the thread pin) twice and with `OMP_NUM_THREADS` = 2 and 8;
  - Linux arm64 (OpenBLAS 0.3.34) in a container, on the working tree;
  - a fresh copy of the repository (a clone of the v7 commit plus the F4sP files) in a Linux arm64
    container with only `requirements-submission.txt` installed;
  - Linux x86-64 under Rosetta emulation (OpenBLAS, Haswell kernels) in a container.

  The containers need the system OpenMP runtime for LightGBM (`apt-get install libgomp1` on
  `python:3.14.6-slim`); without it LightGBM fails to import. Native x86-64 hardware was not tested.
- **Noise probe.** Multiplying the member probabilities by (1 + 1e-6 u), u ~ U(−1, 1), over 3
  draws changes 0 frames; rounding them to 1e-6 changes 0 frames.
- **F4b unchanged.** F4sP only adds files: no code that F4b runs was changed (`git diff` empty),
  the TCN and radar members' match outputs equal those saved by F4b's match run, and re-decoding
  F4b's saved match outputs with F4b's config reproduces its CSV (sha256 `228fe6a4…`). F4b's full
  pipeline was not re-run.
- **Run time (numpy backend):** `make_submission.py` (one thread) took 54 minutes (3230 s wall,
  2757 s CPU) on the Mac while it was heavily shared (load average about 35 on 18 cores); the
  pointer is about 93% of it (3177 of 3423 s in `match_run.py`). Other members: TCN 54 s, MS-TCN
  109 s, radar 81 s, decode 1 s. The Linux arm64 containers took 64-66 minutes. F4b took about 4-7
  minutes.
- **Torch backend** (PyTorch 2.15.0, CPU, float64; the default since 10 Nov 2026). Same CSV sha256
  `60fc793e…` from:
  - macOS arm64: `make_submission.py` (one thread), and `match_run.py` with 1, 2 and 8 threads
    (`OMP_NUM_THREADS` = `VECLIB_MAXIMUM_THREADS` = `MTAH_TORCH_THREADS` = k);
  - Linux arm64 in a container on the working tree (torch 2.15.0+cpu from the PyTorch CPU index,
    BLAS OpenBLAS); Linux x86-64 under Rosetta emulation in a container (torch 2.15.0+cpu);
  - a fresh clone of the repository (commit 5dfbbd3 plus these changes) in a Linux arm64 container
    with only `requirements-submission.txt` (and libgomp1) installed.

  Member level, the shipped pointer run with both backends on identical inputs
  (`$MTAH_WORK/arch/r5_arch/torch_infer/scripts/ptr_cmp.py`):
  - match: max |ΔP| 2.4e-15, max |Δ hand-over prob| 1.7e-15, 0 float32-rounded probabilities and 0
    member codes differ (199,319 frames); both equal the v8 build's saved numpy outputs;
  - DFL J03WN1 seed 2027 under `bench.py --dfl2` (leave-one-game-out model `fold_tcn_1.npz`):
    2.4e-15, 1.9e-15, 0 and 0;
  - Metrica Sample Game 1 seed 2027: 2.2e-15, 1.5e-15, 0 and 0; Sample Game 2 seed 2027: 2.7e-15,
    2.2e-15, 0 and 0.

  The whole of `pipeline.py` with the torch backend, run by the bench with `--save-probs` (DFL
  J03WN1, all 3 seeds, leave-one-game-out models; Metrica Sample Game 1 both seeds and Sample Game
  2 seed 2027), gives probabilities and codes bit-identical to those saved by the live = offline
  check above (numpy backend), and the same bench scores.
  The 1-thread `match_run.py` noise probe (3 draws) and rounding probe changed 0 frames, as with
  numpy. With the new code and `MTAH_PTR_BACKEND=numpy`, `make_submission.py` still writes the same
  file on macOS (30 min 8 s, load average about 8).
- **Run time (torch backend):** `make_submission.py` (one thread) took 3 min 24 s, 5 min 28 s and
  9 min 54 s on the Mac, which other jobs shared (load averages about 5, 8 and 9; the slowest run
  overlapped other jobs' heavy numerical work). The pointer's forward pass alone took 83 s for the
  match (vs 1632 s for numpy in the same comparison run). In `match_run.py` with 1, 2 and 8 threads
  (load 8-20) the pointer member took 189, 191 and 296 s, the whole run 251, 257 and 509 s; more torch
  threads did not help (Accelerate's matrix products). Linux containers (6 virtual CPUs, shared
  Mac): arm64 14 min 55 s (working tree) and 15 min 36 s (fresh clone), x86-64 under Rosetta
  24 min 0 s.

## Files

| file | role |
|---|---|
| `pipeline.py` | the submission pipeline (bench interface: `predict`, `predict_proba`) |
| `live_vs_offline.py` | live codes vs the offline decode of saved member probabilities (and, with `--members-dir`, member by member) |
| `dump_members.py` | bench wrapper for that check: runs `pipeline.py` unchanged and saves each member's live outputs |
| `match_run.py` | label-free BAR-ATH run: timing, statistics, sha256, noise and rounding probes |

Needed beside this folder: `../mbr_decode/`, `../adapt/`, `../chunked_weights.py`, `../nn_pointer/` (with
`pipeline_ballc_p20m.py`, `ptr_torch.py` + `ptr_ballc_p20m.npz`), `../nn_pff/pipeline_c192ls_skcft.py` +
`mstcn_c192ls_skcft.npz`, `../nn_pff/pipeline_w.py`, `../nn_pff_tcn/pipeline_ls.py` + `tcn_ls.npz`, and
the folders v6 already uses (`ensemble_ctrl_v6`, `nn_pff`, `nn_pff_tcn`, `ball_radar_pff`,
`learned_dfl_evidence`).
