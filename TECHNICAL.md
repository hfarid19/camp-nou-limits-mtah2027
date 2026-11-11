# Ball Possession Prediction Using Tracking Data and Open Labelled Games

Team Camp Nou Limits, More than a Hack 2027, Phase 1 (ball possession).

Our final submission is version 8 (`candidates/final_f4sp`). Running `python make_submission.py` regenerates it byte for byte: `evaluation/results/camp_nou_limits.csv`, sha256 `60fc793e90f1df6ce5dda4b2140988fdfd1c1cfc4ee01950b10c429186dd6c57`, 326 possessions, 45.7% dead ball.

This document is the summary. The full record of every experiment, decision rule and check is in [`docs/TECHNICAL_detailed.md`](docs/TECHNICAL_detailed.md).

## 1. Motivation

The challenge was to label each of the 199,319 frames of the Barcelona vs Athletic match as home possession (H), away possession (A), dead ball (D) or not graded (X). The submission is ranked on spellF1, which rewards finding whole possessions with the right team and the right start and end, and macroF1 is the tie-break.

We sought to solve this without ever labelling the provided match. The rules give us 5 public minutes of truth, which is far too little to train on, and the rules also say it must not be used for training or tuning. However, there is a wealth of open football tracking data from other games that already carries possession labels. Therefore, our approach was to learn possession from those other games and make their clean tracking look like the automatic tracking of the provided match.

We also sought to take our analysis a step further and understand how much model size matters for this task. We trained models from 0.1 million to 1 billion parameters and compared the results with two published studies of scaling laws. That analysis is in section 3g.

![How the pipeline works](docs/fig/pipeline.png)

## 2. Dataset

### 2a. The Provided Match

The provided match gave us player and ball tracking at 25 frames per second, Smart Tagging events and video. We used the tracking as the main input and the Smart Tagging tags for small priors, such as marking every frame before a kick-off as dead ball. The submission does not use the video or the homography, since the tracking is already in pitch coordinates.

The only things we took from the match during development were statistics that need no labels. For example, we measured how long the ball goes missing and how smooth its path is. The ball is missing on 41% of frames, in about 127 long gaps per 90 minutes with a median of 10.7 seconds.

### 2b. Labelled Games From Other Sources

| Dataset | Games | Where the labels come from | How we used it |
|---|---|---|---|
| DFL / IDSSE Bundesliga open data (CC BY 4.0) | 7 | The league's official ball status and ball possession flags, per frame | Training and our main benchmark |
| PFF FC World Cup 2022 free release | 64 | Our script, from PFF's own on-ball and out-of-play events | 48 for training, 16 held out and scored once |
| SkillCorner A-League open data (MIT) | 20 | Our script, from SkillCorner's possession events | Benchmark, and fine-tuning data for one model |
| Metrica Sports sample games | 3 | Our script, from Metrica's events | Evaluation only, never trained on |
| FOOTPASS / SoccerNet (under a non-disclosure agreement) | 51 | Our script, from its on-ball events | One dropped training test and a 13-game holdout scored once, locally |
| SoccerTrack v2 (CC BY 4.0) | 10 | Ball action events in the dataset | Evaluation of a dropped video experiment |

No person on our team labelled anything by hand. The Bundesliga labels are the most trustworthy because the league's tracking provider produced them. The other labels come from our scripts applied to each provider's events.

One issue we faced was that the Bundesliga flag changes on every touch, while the competition counts a change of possession only when the opponent gains controlled possession. To fix this we scored against the Bundesliga labels with short opponent spells folded back into the surrounding possession. We then needed to know how short "short" is. Four independent sources agreed that the truth counts possessions finely, with a threshold of about 1 to 1.5 seconds:

| Source | Possessions per 90 minutes |
|---|---|
| Bundesliga flag, raw | 412 |
| Bundesliga flag, 1 second fold | 345 |
| Metrica's own events on its sample games | 305 to 331 |
| PFF World Cup labels | 314 |
| Bundesliga flag, 1.5 second fold | 303 |
| SkillCorner labels | 279 |
| Bundesliga flag, 2 second fold | 272 |

Metrica co-organises the challenge and produced its tracking, so its count carried the most weight. The shift scores that the organisers publish in EVALUATION.md point the same way. Our first submission was built for a 2 second threshold, and moving to the finer reading was the largest single correction we made.

### 2c. Making Clean Tracking Look Like the Match

The open games have clean tracking and the provided match has automatic tracking with gaps. Therefore, we built a simulator that degrades the clean games. Our first simulator hid the ball in many short random bursts. After examining the match statistics we realized this was wrong, because the real ball disappears in a small number of long gaps and the players move during those gaps exactly as they do at dead ball. We rebuilt the simulator and fitted its settings to the match statistics. Each training game is degraded under 3 random seeds.

We also discovered a bug late in the project. For one of the 7 Bundesliga games the attack direction was mirrored in the training data. Fixing it gained +0.008 on the Bundesliga benchmark.

Our final training set for each model was 43 games per fold (36 World Cup and 7 Bundesliga), which is about 1.3 million unique labelled time steps and about 13,000 possessions.

## 3. Models and Analytics

### 3a. How We Score

Every result below is leave-one-game-out. A game is only ever scored by models that never saw it during training, and the benchmark code refuses to run if that is violated. Our selection score is spellF1 averaged over fine readings of the labels (the raw flag and folds of 0.5 and 1 second).

We used four benchmarks:

| Benchmark | Games | Role |
|---|---|---|
| Bundesliga | 7 games, 3 degradation seeds each | Main benchmark, official labels |
| SkillCorner | 20 | Second selection benchmark, different league and label style |
| World Cup, out of fold | 48 | Screen for new models |
| Metrica | 2 | Safety check |

Differences are reported as mean ± standard error over games, with the number of games won in brackets.

### 3b. From Hand-Written Rules to Learned Models

The original pipeline used hand-written evidence, such as the nearest player to the ball, decoded with a 3-state Viterbi algorithm. Its spellF1 on the Bundesliga benchmark was 0.36.

When we replaced the hand-written evidence with gradient boosted trees on about 190 features per frame, the score rose to 0.62. This is a substantial increase and it told us that the evidence was the weak part.

We then moved to sequence models that read the whole game. The four models in the final submission are:

| Model | Parameters | What it does |
|---|---|---|
| TCN | 0.2M | A temporal convolutional network over the per-frame features, with about 108 seconds of context. At prediction time it averages over 4 slightly jittered copies of the input and adapts its normalisation to the game. |
| MS-TCN | 2.6M | A multi-stage temporal network that also predicts where possessions start and end. Fine-tuned on the 20 SkillCorner games. |
| Ball radar | 0.1M | A network that infers where the missing ball is and who has it from player movement alone, followed by gradient boosted trees. |
| Pointer | 19.3M | For each frame it looks at the 8 players nearest the ball and points at the one in possession, or at "dead ball". |

The pointer is our best single model, with 0.764 on the Bundesliga benchmark alone.

### 3c. The Possession Rules

- **Dead ball.** A frame is D wherever the models believe the ball is out of play. Every frame before each half's kick-off is D.
- **Loose balls and passes in flight** stay with the last team in possession.
- **Controlled possession.** An opponent spell shorter than 1.25 seconds that directly follows the other team's possession is handed back. A touch or a deflection without control does not change possession.
- **After a stoppage** the first possession is kept even if it is short, since a restart always has a clear possessor.

The 1.25 second threshold sits between what the Bundesliga benchmark prefers (1 second) and what SkillCorner prefers (1.5 seconds).

### 3d. Combining the Models

Our first combination was a majority vote. A vote over all 11 of our early models was worse than the best single model. The reason was that 7 of them shared the same evidence and made the same mistakes, so they outvoted the diverse ones. Selecting models with nested leave-one-game-out picked exactly the three diverse ones, and that vote became version 6.

Based on this, we decided to improve the combiner itself. The final submission uses minimum Bayes risk (MBR) decoding. For each stretch of play it considers several complete labellings, such as the vote under different smoothing rules and each model alone. It then keeps the labelling with the highest expected spellF1 against samples drawn from the models' probabilities. This optimises the competition metric directly.

| Step | Bundesliga, vs version 6 | SkillCorner, vs version 6 |
|---|---|---|
| Label smoothing in the TCN and a wider MS-TCN | +0.0099 ± 0.0021 (17/21) | +0.0054 ± 0.0026 |
| MBR decoding in place of the vote | +0.0137 ± 0.0044 (16/21) | +0.0123 ± 0.0033 |
| Test-time averaging and adaptation for the TCN | +0.0195 ± 0.0047 (20/21) | +0.0178 ± 0.0035 |
| Pointer added as a fourth model (version 7) | +0.0269 ± 0.0032 (20/21) | +0.0186 ± 0.0040 (17/20) |
| SkillCorner fine-tuned MS-TCN and 19.3M pointer (version 8) | +0.0330 ± 0.0038 (21/21) | +0.0321 ± 0.0039 (20/20) |

### 3e. Fine-Tuning on SkillCorner and a Larger Pointer

After version 7 we ran two more experiments. Each had its decision rule written down before the first result.

The first added the 20 SkillCorner games to training. We tried a full retrain and a short fine-tune, for the MS-TCN and for the pointer. SkillCorner scores were cross-validated so that each game was scored by a model that had not trained on it.

| Variant, compared with its parent | SkillCorner | Bundesliga | Result |
|---|---|---|---|
| MS-TCN, fine-tuned | +0.0388 ± 0.0026 (20/20) | +0.0054 ± 0.0019 | Passed |
| MS-TCN, retrained | +0.0869 ± 0.0047 (20/20) | −0.0087 ± 0.0053 | Failed on Bundesliga |
| Pointer, fine-tuned | +0.0302 ± 0.0029 (20/20) | −0.0099 ± 0.0030 | Failed on Bundesliga |
| Pointer, retrained | +0.0679 ± 0.0051 (20/20) | −0.0154 ± 0.0054 | Failed on Bundesliga |

Only the fine-tuned MS-TCN improved SkillCorner while keeping its Bundesliga score. The other three learned to predict fewer and longer possessions, which suits SkillCorner's coarser labels and costs on the Bundesliga's finer ones.

The second experiment widened the pointer. On the 48 World Cup games the score rose from 0.8137 at 0.8M parameters to 0.8270 at 19.3M, then fell slightly at 58.8M. The 19.3M pointer passed its checks narrowly (Bundesliga +0.0084 ± 0.0039) and went into version 8.

### 3f. Comparison of Versions

![Score of each version on the four benchmarks](docs/fig/progress.png)

Each cell is spellF1 / macroF1.

| Version | What it is | Bundesliga (7) | SkillCorner (20) | World Cup (48) | Metrica (2) |
|---|---|---|---|---|---|
| **8 (final)** | Version 7 with the fine-tuned MS-TCN and the 19.3M pointer | **0.7615 / 0.941** | **0.6285 / 0.901** | **0.824 / 0.950** | **0.7462 / 0.834** |
| 7 | MBR decoding of TCN, MS-TCN, radar and pointer | 0.7555 / 0.940 | 0.6150 / 0.896 | 0.819 / 0.950 | 0.7416 / 0.834 |
| 6 | Majority vote of three models, trained on Bundesliga and World Cup | 0.7286 / 0.933 | 0.5964 / 0.892 | 0.782 / 0.943 | 0.7285 / 0.831 |
| 5c | The same vote, trained on Bundesliga only, after the data fix | 0.717 / 0.930 | 0.594 / 0.889 | 0.682 / 0.907 | 0.697 / 0.825 |
| 4 | First version built for the fine reading of "controlled" | 0.710 / 0.927 | 0.590 / 0.888 | 0.676 / 0.906 | 0.708 / 0.825 |
| 2 (first) | Vote built for a 2 second threshold | 0.662 / 0.923 | 0.611 / 0.891 | 0.632 / 0.902 | 0.664 / 0.822 |

Version 8 is the best version on every benchmark. Against version 7 it gains +0.0060 ± 0.0020 on the Bundesliga (17/21), +0.0135 ± 0.0019 on SkillCorner (17/20) and +0.0046 on Metrica. The World Cup figures for versions 7 and 8 are approximate, because they were computed without the TCN's test-time averaging.

Version 7 was also confirmed once on 13 FOOTPASS games that no model had been scored on before: 0.8046 against 0.7744 for version 6, a gain of +0.0302 ± 0.0045 that held in all 13 games.

### 3g. Model Size and Scaling Laws

Our final goal was to understand whether larger models would do better. Based on the success of the 19.3M pointer, we trained one model per size for each of our five model families, from the shipped size up to 500 million parameters, and for some families up to 1 billion. Each was trained on the first World Cup fold and scored on its 12 held-out games, on SkillCorner and on Metrica.

Each cell is spellF1 / macroF1. The size used in version 8 is marked with `*`. The best size per family, by the average of the six numbers, is in bold.

| Model | Size | World Cup fold (12) | SkillCorner (20) | Metrica (2) | Average |
|---|---|---|---|---|---|
| Pointer | 0.8M | 0.8083 / 0.949 | 0.5879 / 0.892 | 0.7650 / 0.838 | 0.8067 |
| **Pointer** | **19.3M\*** | **0.8217 / 0.954** | **0.5902 / 0.895** | **0.7616 / 0.837** | **0.8099** |
| Pointer | 178M | 0.8213 / 0.952 | 0.5823 / 0.892 | 0.7604 / 0.837 | 0.8075 |
| Pointer | 498M | 0.8157 / 0.952 | 0.5820 / 0.891 | 0.7599 / 0.836 | 0.8061 |
| TCN | 0.2M* | 0.7903 / 0.941 | 0.5850 / 0.889 | 0.7316 / 0.829 | 0.7943 |
| TCN | 2.1M | 0.7913 / 0.943 | 0.5859 / 0.889 | 0.7308 / 0.830 | 0.7950 |
| TCN | 20M | 0.7951 / 0.944 | 0.5779 / 0.887 | 0.7416 / 0.832 | 0.7963 |
| **TCN** | **199M** | **0.7989 / 0.945** | **0.5837 / 0.889** | **0.7321 / 0.830** | **0.7965** |
| TCN | 502M | 0.7933 / 0.944 | 0.5738 / 0.887 | 0.7366 / 0.829 | 0.7940 |
| MS-TCN | 2.6M* | 0.7944 / 0.945 | 0.5558 / 0.885 | 0.6947 / 0.819 | 0.7823 |
| MS-TCN | 20M | 0.8021 / 0.948 | 0.5556 / 0.884 | 0.6862 / 0.816 | 0.7820 |
| MS-TCN | 58M | 0.7993 / 0.948 | 0.5542 / 0.884 | 0.6906 / 0.823 | 0.7832 |
| **MS-TCN** | **198M** | **0.7988 / 0.948** | **0.5560 / 0.884** | **0.6944 / 0.823** | **0.7840** |
| MS-TCN | 500M | training diverged | | | |
| ASFormer | 3.6M | 0.7840 / 0.942 | 0.5459 / 0.882 | 0.6988 / 0.824 | 0.7795 |
| ASFormer | 32M | 0.7709 / 0.940 | 0.5348 / 0.880 | 0.6926 / 0.819 | 0.7729 |
| ASFormer | 198M | 0.7909 / 0.945 | 0.5419 / 0.883 | 0.6923 / 0.823 | 0.7794 |
| **ASFormer** | **505M** | **0.7975 / 0.947** | **0.5403 / 0.882** | **0.6888 / 0.823** | **0.7798** |
| Radar | 0.1M* | 0.7175 / 0.921 | 0.5848 / 0.886 | 0.7343 / 0.832 | 0.7793 |
| Radar | 1.1M | 0.7364 / 0.926 | 0.5933 / 0.888 | 0.7340 / 0.834 | 0.7853 |
| **Radar** | **11M** | **0.7408 / 0.927** | **0.5959 / 0.888** | **0.7329 / 0.834** | **0.7864** |

The MS-TCN rows are the plain versions, without the SkillCorner fine-tune that the shipped one has. The ASFormer is a transformer family that we tested and did not ship.

![Score against model size on the three benchmarks, and held-out loss](docs/fig/size_curves.png)

Looking at the chart, we can see three things.

1. **Larger models gain a little on the World Cup games and gain nothing on SkillCorner.** The 199M TCN is +0.0086 ± 0.0039 better than the shipped TCN on the World Cup fold and −0.0013 ± 0.0036 on SkillCorner. The 502M TCN is −0.0113 ± 0.0044 on SkillCorner. The pointer peaks at the shipped 19.3M and is −0.0080 ± 0.0031 on SkillCorner at 178M.
2. **The radar is the exception.** It gains on both benchmarks as it grows: +0.0233 ± 0.0053 on the World Cup fold and +0.0111 ± 0.0022 on SkillCorner at 11M, winning 18 of 20 SkillCorner games. It started as our smallest model, at 0.1M parameters.
3. **Three of the five "best" sizes are ties.** The TCN at 199M beats the 20M by 0.0002 on the average, the ASFormer at 505M beats the 3.6M by 0.0003, and all MS-TCN sizes sit within 0.002.

**Double descent.** The ASFormer was the only family that logged a held-out loss during training, and it shows a clear double descent. The loss rises from 0.1462 at 3.6M to 0.1595 at 32M. It then falls to 0.1415 at 198M and 0.1368 at 505M, below every smaller model. The spellF1 on the World Cup fold follows the same shape: 0.7840, then 0.7709, then 0.7909 and 0.7975. On all 48 World Cup games the 198M ASFormer beats the 32M by +0.0242 ± 0.0019, winning 45 games.

![Double descent in the ASFormer's held-out loss](docs/fig/dd_loss.png)

**Why the bigger models did not help.** Two published studies of scaling explain most of what we saw.

Kaplan et al. (2020), *Scaling Laws for Neural Language Models*, found that a model's loss improves smoothly with size only while the dataset grows with it. With a fixed dataset the improvement stops and the model begins to overfit. Their rule is that an 8 times larger model needs about 5 times more data. We grew our models by a factor of 5,000, from 0.2 million to 1 billion parameters, and the dataset stayed at 43 games and 1.3 million labelled time steps. Therefore, every large model in the table is deep in the data-limited regime, and the flat curves are what the study predicts.

| | Kaplan et al. | Our experiment |
|---|---|---|
| Model sizes | 768 to 1.5 billion parameters | 0.1 million to 1 billion parameters |
| Dataset | 22 billion tokens, scaled with the model | 1.3 million labelled time steps, fixed |
| Passes over the data | About 1 | 30 to 40 |
| Result as size grows | Loss keeps falling | Score is flat or falls on new data |

Hernandez et al. (2022), *Scaling Laws and Interpretability of Learning from Repeated Data*, studied what happens when training data is repeated. They found that repetition causes double descent, that the damage appears at a range of model sizes that depends on how often the data is repeated, and that it hurts performance on data unlike the training set the most. Our training data is almost entirely repetition: every degraded sequence is seen 30 to 40 times. The ASFormer's dip between 8M and 32M parameters and its recovery above 198M match that description, and SkillCorner is our data that is unlike the training set.

One finding of Kaplan et al. did not hold for us. They found that scores on a different distribution improve in step with scores on the training distribution. Our World Cup scores rose with size and our SkillCorner scores stayed flat. We can infer that the gap to SkillCorner comes from differences in the tracking and in how possessions were labelled, which a larger model trained on the same games cannot learn. This agrees with section 3e, where fine-tuning on SkillCorner games moved the SkillCorner score by +0.039 and no change in size moved it by more than +0.011.

The practical conclusion is that more labelled games would help this task more than more parameters. The 19.3M pointer was the largest model that earned its place in the submission.

Following these two studies, we started three more experiments on the 0.2M and 20M TCN and the 2.6M and 20M MS-TCN: training with fewer and more passes over the data, training on a quarter and a half of the games, and training each recipe with three random seeds. Each size in the table above was trained once, so the seed experiment will show how much of a 0.005 difference is run-to-run noise. Their results will be added to the detailed document.

### 3h. What We Tried and Dropped

| Idea | Result | Why we dropped it |
|---|---|---|
| A vote over all 11 early models | Worse than the best single model | Correlated models outvoted the diverse ones |
| Semi-Markov decoding, learned switch costs, pitch control features | Within noise of the simpler decoder | No gain for the added complexity |
| A classifier that decides per possession whether to keep or hand back | +0.002 | Possession length alone did as well |
| A synthetic ball built from events as training input | −0.049 (0/21) | The models rely on the real ball's motion |
| A video touch detector on ball-centred crops | −0.0001 to −0.0053 | Its touches did not separate right from wrong short possessions |
| A time-series foundation model, fine-tuned | −0.07 on the World Cup games | Clearly worse than the TCN |
| ASFormer and Temba transformers as a fifth model | +0.0005 and 0.0000 | No value inside the ensemble |
| Votes of 5 to 7 models | −0.0002 to −0.0048 on SkillCorner | No better than 3 to 4 models |
| 8 degradation seeds per game in place of 3 | +0.0107 against +0.0099 | Reproduced the gain and added nothing |
| Fine-tuning on Metrica's sample games | Failed its SkillCorner check | Gains did not transfer |
| A SkillCorner fine-tuned 19.3M pointer inside the ensemble | Bundesliga −0.0077 ± 0.0015, SkillCorner +0.0068 ± 0.0015 | No net gain |
| Models larger than those shipped (section 3g) | Flat or worse on SkillCorner | Data-limited |

The complete list, with about 30 more entries, is in the detailed document.

### 3i. What the Models Learned

To understand what the models rely on, we trained two small decision trees the same way. They score 0.65 to 0.69, below the ensemble, and their rules are readable:

- **Dead ball.** The ball is untracked or has moved less than 0.58 metres in half a second, nobody has controlled it for over 3 seconds, and the players average under 2.1 metres per second.
- **Which team.** The team that last had a player alone on the ball keeps it. This rule alone is right on 86% of held-out frames. The most useful single summary is "which team was near the ball over the last 3 seconds".

The pointer, which looks only at the players nearest the ball, is our best single model, which fits this picture.

## 4. Impact

Our approach shows that ball possession can be predicted on automatic tracking without any labels from the match itself. Open labelled games, degraded to look like the target tracking, were enough to raise spellF1 on our main benchmark from 0.36 for the hand-written pipeline to 0.762 for the final ensemble. This is useful for any club or analyst who has tracking from a new provider and no labels for it.

The degradation simulator and the leave-one-game-out benchmarks can be reused for other tracking tasks. The scaling study gives a practical guide for this kind of data: with about 50 labelled games, models above 20 million parameters gave no benefit on new data, and one extra source of labelled games was worth more than a 100 times larger model.

To expand our analysis, we would add professionally labelled games from more leagues, since our results say data is the limit. We would also build a touch detector accurate enough to tell a controlled reception from a deflection. Our simulation says it would need team accuracy of at least 0.95 to help.

### 4a. Limitations

We cannot say for certain that these results will hold on the hidden match, for the following reasons.

- **The benchmarks are proxies.** Both the degradation and the control threshold are inferred. If the truth were coarser after all, with a 2 second threshold, our move to fine-grained output would cost about 0.04.
- **We reused a small benchmark.** Model weights never see the game they are scored on. However, for three weeks every recipe and rule was chosen on the same 7 Bundesliga games and later on SkillCorner as well, so the absolute scores are optimistic. An independent reviewer's fully nested re-selection of an earlier step kept about half of its gain.
- **Version 8 has no once-only confirmation.** The 16 held-out World Cup games were spent on version 6 and the 13 FOOTPASS games on version 7.
- **SkillCorner is training data for one model in version 8.** Its SkillCorner scores are cross-validated, and they remain the least independent of our numbers.
- **The match looks different from our training games.** A classifier separates the match from every Bundesliga game with an AUC of 0.99. The models also predict fewer possessions on the match (277 per 90 minutes) than on any benchmark (310 on the Bundesliga).
- **One match is a single noisy draw.** The game-to-game spread of spellF1 is about 0.025, which is larger than most of the gains in this document.
- **Short and contested possessions are still missed.** About 27% of true possessions were missed on the Bundesliga benchmark, most often those under 2 seconds.
- **The Smart Tagging priors are unvalidated**, because none of our benchmarks has Smart Tagging. Their effect is small.

### 4b. Disclosures

- **Decision rules.** From round 4 on, the rule for accepting a change was written before the results. The rule files are copied verbatim in `docs/prereg/`. We changed the round 5 rule while round 4 was running, after some results had been seen. The change made the rule more conservative. The recipe that built version 7 was written after some of its parts had already been scored. The pointer was admitted on its second variant, after the first failed its check.
- **Scale of the search.** We compared 74 vote sets, 72 decoder variants, about 40 adaptation variants and 49 architecture screens. At that scale the ranking among the top candidates is noise, which is why the final was built one slot at a time under a written recipe.
- **The public 5 minutes.** They were never used for training, tuning or selection. Finished versions were scored on the window out of curiosity, which on two occasions deviated from our own rule of not looking. The window holds 17 true possessions, so one or two possessions move the score by several points.

| Scored on the public window | spellF1 | macroF1 | Possessions predicted |
|---|---|---|---|
| Version 2 | 0.6571 | 0.8801 | 11 |
| Version 6 | 0.6556 | 0.7986 | 19 |
| Version 7 | 0.5667 | 0.8378 | 19 |
| Version 8 | 0.6211 | 0.8348 | 21 |
| Exploratory ensembles X1, X2, X3 | 0.5829, 0.6182, 0.5943 | 0.8373, 0.8812, 0.8375 | 18, 16, 18 |
| Single TCNs of the size study (0.2M, 2.1M, 20M) | 0.7077, 0.6000, 0.6182 | 0.8671, 0.8161, 0.7958 | 22, 19, 16 |

The newer versions score below version 6 on these 17 possessions while beating it on every large benchmark. We report this as it is. No version was chosen, changed or rejected because of the window.

- **FOOTPASS.** It was used locally only. No FOOTPASS data, labels or weights are in this repository, and no shipped weights were trained on it.
- **GPU scoring in the size study.** The benchmark scores of the larger models were computed on a GPU. Each family's GPU path was checked to give the same labels as the CPU path on sample games. The 178M and 498M pointer rows were scored on the GPU and their CPU checks were not completed.

## 5. Repository Contents

```
.
├── TECHNICAL.md                    # this file
├── docs/TECHNICAL_detailed.md      # the full record of experiments, rules and checks
├── docs/prereg/                    # decision rules, written before the results
├── docs/fig/                       # figures used in this document
├── make_submission.py              # regenerates the submission CSV
├── requirements-submission.txt     # pinned environment for the submission
├── candidates/
│   ├── final_f4sp/                 # version 8, the submission
│   ├── final_f4b/                  # version 7
│   ├── ensemble_ctrl_v6/           # version 6
│   ├── nn_pff_tcn/, nn_pff/        # TCN and MS-TCN
│   ├── ball_radar_pff/             # ball radar
│   ├── nn_pointer/                 # pointer
│   ├── mbr_decode/                 # the MBR decoder
│   ├── bench.py                    # the leave-one-game-out benchmark
│   └── archive/                    # dropped experiments
└── evaluation/                     # the organisers' evaluation code
```

The match data is not committed. It is unpacked into `data/` as described in `DATA.md`.

## 6. How to Reproduce

Install the pinned environment (Python 3.14.6) and run the pipeline:

```sh
pip install -r requirements-submission.txt
python make_submission.py
python evaluation/validate_submission.py evaluation/results/camp_nou_limits.csv
```

The expected sha256 is `60fc793e90f1df6ce5dda4b2140988fdfd1c1cfc4ee01950b10c429186dd6c57`.

(Linux users in a slim container also need `apt-get install libgomp1` for LightGBM.)

The submission loads committed weights only (about 93 MB), needs no network and never touches the ground truth. We obtained exactly this file on:

| Platform | Run time |
|---|---|
| macOS arm64, with 1, 2 and 8 threads | 3.5 to 10 minutes |
| Linux arm64 in a container, including a fresh clone | 15 to 16 minutes |
| Linux x86-64 in a container under emulation | 24 minutes |

The pointer's network runs in PyTorch on the CPU. If PyTorch is missing, a numpy version gives the same file in about 54 minutes. Native x86-64 hardware was not tested. Training used GPUs and is not bit-reproducible, so retraining gives slightly different weights.

Rebuilding the models from the open data is described in `candidates/README.md` and in each candidate's README.

## 7. Data Sources

- [DFL / IDSSE open data](https://doi.org/10.6084/m9.figshare.28196177): Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12. © DFL / IDSSE, CC BY 4.0.
- [PFF FC World Cup 2022](https://fc.pff.com): PFF FC's free tracking and event release.
- [SkillCorner open data](https://github.com/SkillCorner/opendata): open-sourced jointly by [SkillCorner](https://skillcorner.com) and [PySport](https://pysport.org/), MIT licence.
- [Metrica Sports sample data](https://github.com/metrica-sports/sample-data).
- FOOTPASS / SoccerNet PC-BAS 2026: arXiv 2511.16183, access-gated.
- SoccerTrack v2, release v1.2, CC BY 4.0.

References for section 3g:

- Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu and Amodei (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
- Hernandez, Brown, Conerly, DasSarma, Drain, El-Showk, Elhage, Hatfield-Dodds, Henighan, Hume, Johnston, Mann, Olah, Olsson, Amodei, Joseph, Kaplan and McCandlish (2022). *Scaling Laws and Interpretability of Learning from Repeated Data.* arXiv:2205.10487.

The shipped weights file `candidates/nn_pff/mstcn_c192ls_skcft.npz` was trained partly on the SkillCorner open data, so its licence notice is reproduced here:

```
The MIT License

Copyright (c) 2020 SkillCorner. https://skillcorner.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
