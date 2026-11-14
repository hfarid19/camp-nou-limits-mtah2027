# final_f4spm: v9 ("F4sP-MF2"), the submission from 11 to 13 Nov 2026

`pipeline.py` is `../final_f4sp` (F4sP, submission v8) with one member slot swapped: the MS-TCN
`m5c_C192ls_skcft` is replaced by `m5c_C192ls_skcft_metft`, the same network fine-tuned once more
with Metrica's three open sample games added. Everything else is v8's: the other three members, the
minimum-Bayes-risk (expected-spellF1) decoding of `../mbr_decode` with the same decoder setting, and
dead ball before each half's kick-off. Like v8 it is deterministic (fixed sampling seed), rounds
member probabilities to float32 before decoding, and runs on numpy and LightGBM plus PyTorch on the
CPU for the pointer's forward pass.

**Status.** Superseded on 13 Nov 2026 by v11 (`../final_v11`), which `make_submission.py` now runs by default; v9 needs
`--candidate final_f4spm --out <path>.csv`. The rest of this file is as written while v9 was the submission.
From 11 Nov 2026 it was the submission: `make_submission.py` ran it by default
(`DEFAULT_CANDIDATE = "final_f4spm"`) and writes `evaluation/results/camp_nou_limits.csv`. v8
(`candidates/final_f4sp`) stays available with `--candidate final_f4sp --out <path>.csv`. Other ways to run v9:

```sh
python make_submission.py --candidate final_f4spm --out <path>.csv   # --out is required for a non-default candidate
python candidates/final_f4spm/match_run.py --tag <tag> [--probes]    # writes to $MTAH_WORK/arch/r5_arch/final_f4spm/csv/
```

The 11M radar ensemble that the switch was waiting for gave no gain (docs/reports/radar_11M_report.txt), so v9
ships with v8's radar.

| slot | member file | what it is | weights |
|---|---|---|---|
| TCN | `adapt/member_tcn_tta4_a50.py` | `o_ls05`, unchanged from v8 (label-smoothed TCN, test-time averaging over 4 jittered copies, half-way feature renormalisation) | `nn_pff_tcn/tcn_ls.npz` |
| MS-TCN | `nn_pff/pipeline_c192ls_skcft_metft.py` | `m5c_C192ls_skcft_metft` (new): v8's MS-TCN fine-tuned with the 10 Nov MET-FT recipe. Each batch of 8 crops (1024 frames) mixes 1:1 crops of the parent's own training data (7 DFL + 48 PFF + 20 SkillCorner games, drawn with the parent's sampling weights) and crops of Metrica Sample Games 1-3 (strict Metrica-rule labels). 750 iterations, Adam at a constant lr of 5e-5, seed 0, label smoothing 0.05, no clipping or EMA. The parent's feature normalisation is kept (mean, std, feature names and network config are byte-equal to the parent's). Trained on an H100 pod | `nn_pff/mstcn_c192ls_skcft_metft.npz` (10.6 MB, sha256 `a7a242c352e3e7fbe18c748c9ce07373640d932b44b8bcb468f49a254d89edf9`; 78 training games: 48 PFF, 20 SkillCorner, 7 DFL, 3 Metrica) |
| radar | `ball_radar_pff/pipeline.py` | `radar_pff`, unchanged | `radar.npz`, `model_br*.txt` |
| pointer | `nn_pointer/pipeline_ballc_p20m.py` | `r5_ptr_ballc_p20M` (19.3M parameters), unchanged | `nn_pointer/ptr_ballc_p20m.npz` |

Decoder config: `src hmm, alpha 0.1, rho 0.2, margin 0.0`. This is the full-DFL pick of the g1 grid
for this member set (`$MTAH_WORK/arch/r5_arch/skcmetft/final_met_logo_F4sPMF2.txt`), which is the
same as v8's.

The member file follows v8's member pattern: `../nn_pff/pipeline.py` runs under a private module
name with `MODEL_PATH` pointing at the new weights. The weights still load through
`pff_folds.resolve`, which calls `dfl_folds.resolve`. Under the DFL bench, that lookup maps the file's
sha256 prefix to its leave-one-game-out fold models. It fails closed: it raises when the prefix
(`a7a242c352e3`) is not in the bench's `--folds` JSON. The v9 folds JSON is v8's
(`folds_f4sp.json`) plus `a7a242c352e3 -> [skcmetft/m5c_C192ls_skcft_metft, tcn]`. Every
`fold_tcn_<g>` model was checked to exclude DFL game g.

## How it was chosen

This is a short account; the full one is in `$MTAH_WORK/arch/r5_arch/skcmetft/report.txt` (sections
6, 8 and 10) and `NOTES.md` (sections 6-8). The rules are in `../../docs/prereg/round5_prereg.txt`
(addenda of 11 Nov 12:30, 12:50, 12:31 and 12:46).

1. Under the 12:30 / 12:50 addenda, Metrica-inclusive versions of the members were trained. The
   MS-TCN and the pointer failed their member gates; the TCN passed. Training of the radar's Metrica
   variant was stopped before it was scored.
2. **Rule change after the results, at the user's direction** (addendum 12:31). After seeing the
   member results, the user decided that a held-out Metrica gain is worth small losses on DFL and
   SkillCorner ("if our final test is Metrica labelled, that tradeoff is worth it"). New rule for
   replacing F4sP: held-out Metrica (ensemble, leave one game out, strict labels) better on at least
   2 of 3 games with a mean gain of at least +0.005; DFL fine4 not worse by more than 0.005;
   SkillCorner-CV fine3 not worse by more than 0.010; the pipeline reproduces byte-identically. The
   first final under it, F4sP-M (three members swapped), failed (Metrica +0.0020).
3. Addendum 12:46 allowed one exploratory final, **F4sP-MF**: F4sP with its MS-TCN replaced by the
   10 Nov MET-FT MS-TCN, which loses the SkillCorner fine-tune. It passed on Metrica (+0.0123, 3/3)
   and DFL (−0.0020 ± 0.0024) but **failed** on SkillCorner (−0.0107 ± 0.0017).
4. The properly built version, **F4sP-MF2**, applies the MET-FT recipe to v8's shipped SkillCorner
   fine-tuned MS-TCN, with its own fold models. It was trained before F4sP-MF's verdict, scored
   after that verdict had been written down, and is marked "scored after a failed precursor". It
   meets the 12:31 score thresholds:

| bench | v8 (F4sP) | v9 (F4sP-MF2) | difference ± SE (wins) | possessions / 90, v8 -> v9 (truth) |
|---|---|---|---|---|
| DFL fine4, 21 game-seeds, decoder settings nested | 0.7615 | 0.7590 | −0.0026 ± 0.0018 (9/21) | 310.4 -> 308.8 |
| SkillCorner-CV fine3, 20 games | 0.6285 | 0.6270 | −0.0015 ± 0.0016 (9/20) | 285.1 -> 285.0 (279) |
| Metrica, leave one game out, strict, Sample Game 1 | 0.7912 | 0.8120 | +0.0208 | 299 -> 294 (307) |
| Metrica, leave one game out, strict, Sample Game 2 | 0.7883 | 0.7948 | +0.0065 | 277 -> 271 (283) |
| Metrica, leave one game out, strict, Sample Game 3 | 0.7808 | 0.7859 | +0.0052 | 316 -> 312 (324) |
| Metrica mean, strict (practice labels) | 0.7868 | 0.7976 | +0.0108, 3/3 (practice +0.0104, 3/3) | |

The new member alone vs v8's MS-TCN: DFL −0.0069 ± 0.0030 (5/21), SkillCorner-CV −0.0014 ± 0.0020
(10/20), and Metrica leave one game out +0.0662 / +0.0587 / +0.0423 (mean +0.0557, 3/3).

Scoring conventions: on DFL and SkillCorner the MS-TCN slot was scored with fold models (each DFL
game by the model that never saw it; each SkillCorner game by the cross-validation fold model that
never saw it). On Metrica, game k was scored with the `fold_met_<k>` model, which never saw game k;
the other three members used their full models. The shipped full model trained on all 7 DFL, all 20
SkillCorner and all 3 Metrica games.

**Caveats:**
- **The rule changed after results were seen, at the user's direction** (addendum 12:31), and this
  must be disclosed in the write-up. Under the rules in force before it, the Metrica-inclusive
  MS-TCN members failed their gates.
- **Scored after a failed precursor.** F4sP-MF failed on SkillCorner, and F4sP-MF2 was then scored
  as its properly built version. This is one more look at the same benches.
- **Metrica is no longer an independent bench.** Metrica Sample Games 1-3 are now training data for
  this member: the shipped model trained on all three games. The Metrica numbers above are leave one
  game out (3 games, weak evidence), and from now on no Metrica number for v9 or its successors is
  untouched evidence. The Metrica sample games were also used to design and select earlier versions.
- **Learning-rate decision (5e-5).** MET-FT's rule was "0.1 × the parent's peak lr". Applied
  literally to the skcft parent, whose own lr was already a constant 5e-5, it would give 5e-6, a 10×
  weaker Metrica dose than the 10 Nov MET-FT model had. The same absolute lr of 5e-5 was used instead
  (0.1 × the lineage's peak of 5e-4, the value SKC-FT and MET-FT both used). This choice was recorded
  before training and is disclosed here.
- **The member alone is worse on DFL** by 2.3 SE (−0.0069 ± 0.0030); in the ensemble DFL drops by
  0.0026 ± 0.0018 and SkillCorner by 0.0015 ± 0.0016, both inside the 12:31 tolerances.
- **No clean confirmation.** The FOOTPASS holdout was spent on v7 (F4b), and every number above
  comes from benches that were reused for design and selection. Multiplicity: on top of v8's
  comparisons, the skcmetft experiment adds 4 scored Metrica-inclusive members (MS-TCN skcmetft, TCN
  o_ls05_met, pointer p20M_metft, MS-TCN skcft_metft), 1 radar stopped before scoring, and 3 finals
  (F4sP-M, F4sP-MF, F4sP-MF2), plus 3 exploratory single-slot swaps in report.txt section 6.
- **Training-data terms.** The Metrica Sports sample data (github.com/metrica-sports/sample-data) has
  no licence file. Its README asks users to "be responsible with the use of this data" and, "if you
  use it for anything public", to "acknowledge the source". Shipping weights trained on it therefore
  needs at least that acknowledgement (see "Training data" below). SkillCorner: MIT licence, ©
  2020 SkillCorner, with the notice in `../../TECHNICAL.md` ("Data Sources").

## Checks

All checks ran on the Mac (Apple M5 Max) or in local containers, using positions and tags of the
match only and no labels. The work log is `$MTAH_WORK/arch/r5_arch/final_f4spm/NOTES.md`. Unless
noted, every run used the torch backend for the pointer (the default).

- **Live = offline** (`live_vs_offline.py --members-dir`). The live codes of `pipeline.py`, run
  through the benches by `dump_members.py` (which wraps it unchanged), equal the offline F4sP-MF2
  decode of the members' saved outputs with `pipeline.CFG` on all 8 units tried:
  - DFL J03WN1, all 3 degradation seeds. The bench resolved `mstcn_c192ls_skcft_metft.npz` to its
    leave-one-game-out `fold_tcn_1` model, through `folds_f4spm.json`. The offline MS-TCN outputs are
    the ones the scored F4sP-MF2 used.
  - Metrica Sample Game 1 and Sample Game 2, both seeds each.
  - SkillCorner game 1874553.

  On Metrica and SkillCorner the scored F4sP-MF2 used fold models for the MS-TCN, while the pipeline
  loads the full model. There, the offline MS-TCN outputs are the full model's, computed by the
  experiment's own scoring path (`nn_pff/pipeline_cfg.py` with `NN_PFF_MODEL=<full.npz>`), not by
  the pipeline under test.

  Member by member: the TCN, MS-TCN and radar probabilities are identical to the saved ones
  (max |ΔP| = 0). The 19.3M pointer's differ from its saved bench outputs, which came from the
  float32 GPU path, by at most 2.9e-6, as in v8. No member's codes differ on any frame.
- **Match** (`match_run.py`: positions and tags only, no labels):
  - 321 possessions (272.8 per 90 graded minutes), mean 10.8 s, dead share 0.456, no X inside the
    halves (v8: 326, 277.0 per 90, 0.457);
  - 1.66% of graded frames differ from v8 (2,637 frames), 2.51% from v7 (3,985) and 3.78% from v6
    (5,997); no frame outside the graded halves differs;
  - v8 cross-check: the TCN, radar and pointer match outputs equal those saved by v8's build run.
    Re-decoding them with v8's saved MS-TCN outputs gives v8's CSV (`60fc793e…`), so the MS-TCN slot
    is the only difference;
  - CSV sha256 `9926b7c56557a7aedb39597b7c2f52b37c82028d01b394021a360bb3aabcb3e2`;
    `validate_submission.py`: valid.
- **Noise probe.** Multiplying the member probabilities by (1 + 1e-6 u), u ~ U(−1, 1), over 3 draws
  changes 0 frames; rounding them to 1e-6 changes 0 frames; noise then rounding changes 0 frames.
- **Byte-identical CSV** (sha256 above), torch backend:
  - macOS arm64 (Accelerate): `make_submission.py --candidate final_f4spm` (always one thread),
    and `match_run.py` with 1, 2 and 8 threads (`OMP_NUM_THREADS` = `VECLIB_MAXIMUM_THREADS` =
    `OPENBLAS_NUM_THREADS` = `MKL_NUM_THREADS` = `MTAH_TORCH_THREADS` = k);
- Linux arm64 (OpenBLAS; torch 2.15.0+cpu from the PyTorch CPU index) in a `python:3.14.6-slim`
    container with `libgomp1` and only `requirements-submission.txt` installed, on the working tree
    (`make_submission.py --candidate final_f4spm`);
  - a fresh copy in the same kind of container: a local `git clone` of the commit before v9 plus only
    this change's files (`make_submission.py`, `final_f4spm/`, the new member file and weights);
  - Linux x86-64 under Rosetta emulation in the same kind of container (torch BLAS MKL, CPU
    capability AVX2). Native x86-64 hardware was not tested.

  The match data were mounted read-only from the Mac; nothing left it.
- **Numpy backend: not re-verified for v9** (`MTAH_PTR_BACKEND=numpy` was not run, at the user's
  direction). The pointer member and its code are unchanged from v8, where both backends gave the
  same file; the member that changed (the MS-TCN) has only one code path (numpy).
- **v8 unchanged.** v9 only adds files, plus the `--candidate` switch in `make_submission.py`.
  `git diff` of `candidates/final_f4sp/` and of the existing weights is empty.
After the switch was added, `python make_submission.py` (the default) still wrote v8's file,
  sha256 `60fc793e90f1df6ce5dda4b2140988fdfd1c1cfc4ee01950b10c429186dd6c57`. Without `--out`,
  `--candidate final_f4spm` is refused, so it cannot overwrite `evaluation/results/camp_nou_limits.csv`.
- **Run time** (Mac lightly loaded, load average about 3): `make_submission.py --candidate
  final_f4spm` took 124 s on one thread, with a peak resident memory of 5.9 GB in the main process
  plus about 4 GB in the torch worker. `match_run.py` timings: TCN 10 s, MS-TCN 9 s, radar 20 s,
  pointer 72-83 s, decode under 1 s; 111-122 s in all for 1, 2 and 8 threads.
Linux containers (6 virtual CPUs, 9 GiB, sharing the Mac with other jobs), `make_submission.py`
  alone: arm64 8 min 10 s (working tree) and 15 min 47 s (fresh copy, with the Mac busier); x86-64
  under Rosetta 19 min 39 s.

## Training data

The MS-TCN trained on the 7 DFL / IDSSE open games (CC BY 4.0), the 48 PFF FC World Cup 2022 games,
the 20 SkillCorner A-League open-data games (MIT licence, © 2020 SkillCorner) and Metrica Sports'
three open sample games (https://github.com/metrica-sports/sample-data: "If you use it for anything
public, please acknowledge the source"). The TCN, radar and pointer are v8's: 7 DFL + 48 PFF games.
No member trained on FOOTPASS or on the 16 PFF holdout games.

## Files

| file | role |
|---|---|
| `pipeline.py` | the v9 pipeline (bench interface: `predict`, `predict_proba`) |
| `match_run.py` | label-free BAR-ATH run: timing, statistics vs v6 / v7 / v8, v8 cross-check, sha256, noise and rounding probes; never writes under `evaluation/` |
| `live_vs_offline.py` | live codes vs the offline decode of saved member probabilities (and, with `--members-dir`, member by member) |
| `dump_members.py` | bench wrapper for that check: runs `pipeline.py` unchanged and saves each member's live outputs |

Beyond v8's files, this folder needs `../nn_pff/pipeline_c192ls_skcft_metft.py` +
`../nn_pff/mstcn_c192ls_skcft_metft.npz`. Nothing that v8 runs was changed. `make_submission.py`
gained the `--candidate` switch (default `final_f4spm` since 11 Nov).
