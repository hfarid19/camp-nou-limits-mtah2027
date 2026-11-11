# Pre-registration files, rounds 4 and 5

Verbatim copies of the decision rules and the finals' reports as they were written in the work directory
(`$MTAH_WORK/arch/`) during rounds 4 and 5. They contain rules, timestamps and bench scores only: no
match data and no labels. Paths inside them are relative to `$MTAH_WORK` (for example
`arch/round5_prereg.txt`), because that is where they were written. The account of how they were
applied, including where the process deviated from them, is in `../../TECHNICAL.md`, section 6.2.

| file | original location | what it is |
|---|---|---|
| `round4_prereg.txt` | `arch/round4_prereg.txt` | round 4 decision rule (9 Nov), with the 11:05 addendum that replaced the SkillCorner holdout confirmation by the FOOTPASS one |
| `round5_prereg.txt` | `arch/round5_prereg.txt` | round 5 decision rule (9 Nov 20:20), the matched-count addendum (20:24), the final-recipe addenda (10 Nov 01:32 and 02:26), and the addenda written after F4b was fixed: the role-aware pointer (10:35), "+SKC" with SkillCorner as training data (11:20, 11:30, 11:50), SKC19 (12:45), the pointer size ladder (13:30, 14:55) and the transformer ensemble-value screen (14:05). Addenda from 20:25 on were written after F4sP was picked |
| `prereg_adapt.txt` | `arch/r5_adapt/prereg_adapt.txt` | the label-free adaptation variants fixed before any result (track R5-2) |
| `vote_plan.txt` | `arch/r5_arch/vote_plan.txt` | the pointer vote sets declared before the pointer's DFL result (and the coordinator's sets that superseded them) |
| `finals_report.txt` | `arch/r5_arch/finals_report.txt` | the scores of the two finals, F3 and F4b, paired vs v6, vs the then-leader and F4b vs F3 |
| `finals_F4s_report.txt` | `arch/r5_arch/skc_train/finals_F4s_report.txt` | the "+SKC" final, F4s (F4b with the SkillCorner-fine-tuned MS-TCN), paired vs v6 and vs F4b |
| `finals_F4sP_report.txt` | `arch/r5_arch/ptr_size/finals_F4sP_report.txt` | the pointer-ladder final, F4sP (F4s with the 19.3M pointer; the submission), paired vs v6, vs F4b and vs F4s |

The timestamps are the authors' own; the files were not hashed or published when written, so they are
a record kept in good faith, not a third-party proof of timing.
