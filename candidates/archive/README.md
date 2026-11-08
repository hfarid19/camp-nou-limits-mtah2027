# Archived candidates

These candidates were built, benchmarked, and not shipped. They are kept as a record of what was
tried and why it was dropped. Each folder's README has the method, references, every variant
tried, and rebuild steps.

They were developed as `candidates/<slug>/`, and most compute the repository root by counting
directory levels. To run one, move it back up one level first (`mv archive/<slug> ..`).

Large model files and tuning caches are not committed (see the repository `.gitignore`); the
READMEs list the commands that rebuild them from the open DFL / IDSSE data.

Scores are spellF1. DFL is leave-one-game-out on the 7 open DFL / IDSSE games, so every game is
scored by models that never saw it; it was the primary selection criterion. Metrica dev is Sample
Games 1 + 2 and the holdout is Sample Game 3, both degraded with `tune_on_sample_data.degrade`.

| slug | idea | DFL | dev | holdout | why it was not shipped |
|---|---|---|---|---|---|
| v1_original | the original `possession_pipeline.py` behind the bench interface | 0.360 | 0.475 | 0.506 | baseline |
| v2_stacked_fixes | v1 with stacked fixes + DFL-trained evidence | 0.617 | 0.676 | 0.682 | superseded by learned_dfl_evidence |
| hier_touch_decoder | hierarchical dead / team decoder with touch snapping | 0.599 | 0.663 | 0.679 | superseded by learned_dfl_evidence |
| pathcrf_dynamic | linear-chain CRF with learned, frame-dependent transition costs (after PathCRF) | 0.625 | 0.665 | 0.681 | parity with learned_dfl_evidence; errors highly correlated with it |
| semimarkov_lattice | explicit-duration segmental decoding over a touch lattice | 0.642 | 0.675 | 0.684 | gain mostly from an evidence tweak; not picked by the ensemble's nested selection |
| ballfree_dead | dead-ball / set-piece model from player positions only | 0.628 | 0.669 | 0.671 | the ball-based dead model is already at dead F1 0.98 on DFL |
| pitch_control | Spearman / Fernandez-Bornn pitch-control features added to the LightGBM evidence | 0.629 | 0.678 | 0.680 | inside noise; does not reduce under-splitting |
| skillcorner_domain | SkillCorner open broadcast tracking as a second training domain | 0.619 | 0.685 | 0.683 | its possession labels change team less often than DFL's, which worsened under-splitting |

The shipped submission is `candidates/ensemble_vote`: a majority vote of `temporal_tcn`,
`handover_mstcn` and `ball_radar` (DFL 0.662 / dev 0.707 / holdout 0.705). A vote over all 11
candidates scored only 0.630 on DFL, because the seven learned_dfl_evidence-like members make
correlated errors and outvote the three diverse ones.
