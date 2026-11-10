# nn_asformer: temporal-transformer / state-space segmentation models on PFF + DFL

First attention-over-time (and state-space) members for the possession task. Everything else is
the team's existing protocol: the NN trainer's dataset (`$MTAH_WORK/ds_pff`, 48 PFF selection
games x 3 degradations + 7 DFL games x 4, kick-off attack sign, 5 fps features of
`learned_dfl_evidence/features.py`), its 16-game PFF holdout (`arch/nn_pff/holdout_games.json`,
scored once at the end), its 4 game-grouped PFF selection folds (`arch/nn_pff/selection_folds.json`)
and DFL leave-one-game-out fold models (`bench.py --dfl2`), and the MS-TCN member's outputs and
ASRF decoder (`nn_pff/pipeline.py`, copied here with the network swapped). Reference: `ensemble_ctrl_v5c`.

RESULTS: see the tables below (filled as the runs finish; every number in this file comes from a
log or a saved codes directory named next to it).

## Models (`model.py`, torch for training only)

One skeleton for three mixers. Encoder stage on the 194 normalised 5 fps features, then S = 2
refinement stages (ASFormer decoders / MS-TCN refinement) fed softmax(class) ++ sigmoid(boundary)
of the previous stage; every stage ends in a LayerNorm and two linear heads (3-class H/A/D logits,
2 boundary logits H<->A and alive<->dead, the ASRF pair). Block i of a stage (pre-norm, residual):

    h = x + DropPath(Dropout(ConvFF(LN(x))))     ConvFF = dilated conv k3 (dilation 2^i) -> GELU -> 1x1
    h = h + DropPath(Dropout(alpha * Mixer(LN(h))))   alpha: learnable scalar gate (ASFormer), init 1

| arch | mixer | positions | decoder stages |
|---|---|---|---|
| `asformer` | local windowed multi-head self-attention, half-width w_i = min(2^i, 256) (window doubling per layer, Yi et al. BMVC 2021); exact band, computed in query chunks | learned relative bias per head over -w..w (window-local, so crops and full halves behave identically) | cross-attention (q from the decoder, k/v from the encoder), same window |
| `temba` | bidirectional Mamba-2 style selective SSM (SSD: scalar decay per head, chunked algorithm; Dao & Gu 2024) after the dilated conv, in the spirit of MS-Temba (arXiv 2501.06138: dilated multi-scale convs + Mamba); own implementation from the papers | none needed (recurrent) | same blocks, no cross-attention |
| `global` | full self-attention over the whole sequence (torch `scaled_dot_product_attention` in training) | ALiBi (fixed per-head linear distance bias; length-agnostic) | global cross-attention |

Recipe (`train.py`, shared): crops of 2048 steps (6.8 min at 5 fps), batch 4, 150 iterations per
epoch, 30 epochs max; team-swap augmentation p = 0.5 per crop (`features.swap_index`); loss per
stage = CE + 0.15 x truncated MSE on log-probs (tau 4) + boundary BCE (pos_weight 10); AdamW
(betas 0.9/0.98, weight decay 0.05 on weight matrices and conv kernels only), linear warm-up (5%)
+ cosine decay, gradient-norm clip 1.0, dropout 0.3, stochastic depth up to 0.1 (linear in depth),
EMA of the weights (0.999; exported), float32 throughout (no mixed precision on MPS). Early stopping
/ model selection on a TRAINING-side validation split: 4 PFF games drawn (seeded) from the spec's own
training games and removed from training; the epoch with the lowest EMA-model class CE on them is
exported (patience 8). The selection fold, the DFL game left out and the holdout are never used
for it. Per-epoch checkpoint / resume. Seeds never 2027-2029 (seed 0).

Inference (`asf_np.py`, numpy float64, deterministic): same chunked band attention / chunked SSD /
chunked global attention; mirror averaging (features team-swapped, H/A outputs swapped back) as the
existing members; the five 5 fps phases interleaved back to 25 fps; decoder = `nn_pff/pipeline.py`'s
ASRF segmentation with its constants (chosen for nn_pff / handover_mstcn_v3; a nested re-choice is
reported below). Parity numpy vs torch CPU: `verify_parity.py` (table below).

## Files

| file | role |
|---|---|
| `model.py`, `train.py` | torch models and trainer (`--device mps`), export to npz |
| `asf_np.py` | numpy forward pass |
| `pipeline.py` | bench interface (predict / predict_proba / boundary_proba); = `nn_pff/pipeline.py` with `asf_np.Model`; weights `asformer.npz` through `pff_folds.resolve` |
| `pipeline_cfg.py` | bench wrapper for a configuration (`NN_ASF_MODEL`) |
| `run_pff.py`, `vote_offline.py`, `decode_sweep.py` | nn_pff's runners / offline tools with the output root `probs/nn_asformer` |
| `report.py` | size x score table (params, train loss, val CE, PFF OOF fine3 vs v5c) |
| `score_dev.py` | Metrica dev paired scorer of saved codes |
| `verify_parity.py` | numpy vs torch parity on full games |
| `chain.sh` | live checks as fold models appear (PFF OOF, then bench --dfl2 with the DFL fold models) |
| `features.py`, `pff_folds.py` | byte-identical copies of nn_pff's |

Fold models, logs and checkpoints: `$MTAH_WORK/arch/nn_asformer/<config>/`; outputs:
`$MTAH_WORK/probs/nn_asformer/<config>` (PFF OOF codes / probs), `probs/nn_asformer_dfl/<config>`
(DFL LOGO), `probs/metrica_dev/<config>`.

## Results

(filled in as runs complete)
