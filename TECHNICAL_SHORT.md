# Ball Possession Prediction Using Tracking Data and Open Labelled Games

Team Camp Nou Limits, More than a Hack 2027, Phase 1. This is the short document. The full write-up is [`TECHNICAL.md`](TECHNICAL.md) in the repository.

## 1. Approach

We sought to label every frame of the match without ever labelling the provided match. Open tracking data from other games already carries possession labels. Therefore, we learned possession from those games and degraded their clean tracking until it looked like the automatic tracking of the provided match. The degradation was fitted to statistics of the match that need no labels.

![How the pipeline works](docs/fig/pipeline.png)

Four models read each game: a temporal convolutional network (TCN), a multi-stage network that also finds possession boundaries (MS-TCN), a ball radar that infers the missing ball from player movement, and a pointer that picks the possessor among the 8 players nearest the ball. A stacker, which is a small model trained on Metrica's sample games, reads the outputs of the four. A minimum Bayes risk (MBR) decoder combines all five. It keeps the labelling with the highest expected spellF1, which is the competition metric.

![First and final submission](docs/fig/short_first_final.png)

## 2. Training Data

Nothing comes from the provided game's labels, and no person on our team labelled anything by hand. Every score in this document comes from models that never saw the scored game.

| Dataset | Games | Labels | Use |
|---|---|---|---|
| DFL / IDSSE Bundesliga (CC BY 4.0) | 7 | The league's official possession flags | Training, main benchmark |
| PFF FC World Cup 2022 | 64 | Our script, from PFF's events | 48 training, 16 held out |
| SkillCorner A-League (MIT) | 20 | Our script, from SkillCorner's events | Benchmark, fine-tuning one model |
| Metrica Sports sample games | 3 | Our script, from Metrica's events | Benchmark, fine-tuning one model, training the stacker |
| FOOTPASS / SoccerNet | 51 | Our script, from its events | 13-game holdout, scored once, local only |

## 3. Assets Used and Dropped

We used the tactical tracking as the main input and the Smart Tagging tags for small priors, such as dead ball before each kick-off. The submission does not use the video or the broadcast feed.

**The broadcast feed.** We aligned the broadcast tracking with the tactical tracking without labels, with one constant time offset per half. It sees the ball on 29% of the aligned frames where the tactical ball is missing. No benchmark has a second feed, so we simulated one on the benchmark games and tested fusion rules there, under a rule written beforehand. None of the 6 rules we ran passed. On the World Cup games a dead-ball prior while the broadcast cuts away cost −0.0017 ± 0.0004, filling every ball gap −0.0072 ± 0.0011, and filling only the gaps that look like live play −0.0008 ± 0.0005. The most careful rule, which fills only gaps the broadcast ball bridges cleanly, passed on the Bundesliga (+0.0002 ± 0.0001) and then lost on SkillCorner (−0.0005 ± 0.0004). It changes only 1 to 5 seconds of ball per 90 minutes.

![Ideas we tested and dropped](docs/fig/short_dropped.png)

The 11M ball radar was clearly better alone and added nothing to the ensemble, because the other three models already covered what it knew. The middle seven bars are from 13 Nov. A spatio-temporal attention model lost to the pointer on SkillCorner (−0.0112 ± 0.0019). An input flag for the provider's labelling style, set to Metrica, lost 0.068 on a World Cup fold and 0.046 on SkillCorner. Acceleration inputs hurt the pointer on SkillCorner (−0.0030) and Metrica (−0.0044). Averaging three seeds per model gave the ensemble +0.0002 on the World Cup fold, +0.0048 on SkillCorner and −0.0001 on Metrica. More augmentation of the Metrica fine-tune changed nothing (−0.0007 ± 0.0008). Five variants of the stacker failed their rule, for example −0.0023 on held-out Metrica when trained with other providers' games. The lowest three bars are from 14 Nov. A training ball like the match's did not recover version 12's SkillCorner loss. Counting the pointer twice in the decoding lost on SkillCorner and Metrica. Counting the stacker twice gained only on Metrica (+0.0090 ± 0.0042, 3 games), one of 64 comparisons, so we left it out. Adding the attention model gained, but adding a second pointer gained almost as much. No combination of the version 9 and 11 models beat version 11's four.

## 4. Possession Rules

- **Dead ball** wherever the models believe the ball is out of play, and before each kick-off.
- **Loose balls and passes in flight** stay with the last team in possession.
- **Controlled possession.** An opponent possession shorter than 1.25 seconds is handed back, since a touch without control does not change possession.
- **After a stoppage** the first possession is kept even if it is short.

The Bundesliga flag changes on every touch, so we needed to know how finely the truth counts possessions. Metrica's and PFF's counts sit between a 1 second and a 1.5 second threshold. Our first submission assumed 2 seconds, and correcting this was our largest single gain.

![Possession counts by provider](docs/fig/short_counts.png)

## 5. Results

**The decoder.** On the four models of version 9, MBR decoding beat a plain vote on every Bundesliga game.

![Decoder against vote](docs/fig/short_decoder.png)

**The Metrica step (version 9).** Metrica co-organises the challenge and produced the match's tracking and tagging, so we expect the hidden truth to follow Metrica's way of labelling. We fine-tuned the MS-TCN on Metrica's three sample games and accepted a small cost on the other benchmarks.

![The Metrica trade-off](docs/fig/short_tradeoff.png)

**The final steps (versions 11 and 12).** Version 11 retrained the TCN and the MS-TCN with 29 more input columns (ball and player acceleration, the ball's turn angle, team shape) and added a stacker as a fifth member: gradient boosted trees on the four models' probabilities at 7 time offsets, trained only on the 3 Metrica games. It was published on 14 Nov and then replaced. Version 12 retrains the TCN and the MS-TCN with 58 more columns: how fast the ball and each team's nearest player close in, detected touches, the ball's jerk and its turns over several time spans, and 13 hand-picked products. The stacker is refitted and the rest is unchanged. Held-out Metrica is leave-one-game-out for the stacker and the MS-TCN.

| spellF1 | Metrica, held out (3) | Bundesliga (7) | SkillCorner (20) | World Cup (48) |
|---|---|---|---|---|
| Version 9 | 0.7976 | 0.7590 | 0.6270 | 0.8227 |
| Version 11 | 0.8155 | 0.7673 | 0.6335 | 0.8319 |
| Version 12 without the stacker | 0.8139 | 0.7717 | 0.6289 | 0.8384 |
| **Version 12** | **0.8197** | **0.7735** | **0.6333** | **0.8413** |
| Version 11 − version 9 (games won) | +0.0179 ± 0.0063 (3/3) | +0.0083 ± 0.0024 (17/21) | +0.0065 ± 0.0020 (16/20) | +0.0093 ± 0.0016 (39/48) |
| Version 12 − version 11 (games won) | +0.0042 ± 0.0024 (2/3) | +0.0062 ± 0.0014 (18/21) | −0.0002 ± 0.0025 (10/20) | +0.0094 ± 0.0013 (42/48) |

![Versions 9, 11 and 12 on the four benchmarks](docs/fig/short_versions.png)

**How version 12's columns were chosen.** We screened each group on one World Cup fold, SkillCorner and two Metrica games. In the TCN every group gained on the World Cup fold and Metrica and lost on SkillCorner. Our rule kept such groups out. After seeing these results we changed it: a group now qualifies if it gains on two of the three benchmarks, preferably including Metrica. Version 12 follows the changed rule, so its columns were chosen on the benchmarks that then scored it.

**Model size.** We trained every model family at sizes up to 500 million parameters. This matches published scaling laws: with a fixed dataset, loss stops improving as models grow (Kaplan et al., 2020), and repeated training data makes it worse before it gets better (Hernandez et al., 2022). With about 50 labelled games, data is our limit.

![Score against model size](docs/fig/short_size.png)

We also saw double descent. In a transformer family that we did not ship (ASFormer), held-out loss got worse up to 32 million parameters and then fell below every smaller model at 198 and 505 million. Only the steps above 32 million changed width alone, so the fall is the clean part of this result. It did not carry over to SkillCorner.

![Double descent in held-out loss](docs/fig/short_dd.png)

## 6. Limitations

- **The benchmarks are proxies.** If the truth counts possessions more coarsely than we believe, our output would cost about 0.04.
- **We reused small benchmarks** for every choice, so the scores above are optimistic. Versions 11 and 12 have no once-only confirmation. An earlier version gained +0.030 on 13 fresh games.
- **Version 12's columns were chosen under a rule changed after results**, on the benchmarks that then scored it. It did not improve SkillCorner (−0.0002 ± 0.0025), the benchmark closest to the match in its ball statistics. Without the stacker its models lose there (−0.0046 ± 0.0022).
- **Metrica is three games**, and the rule for the Metrica step was changed after its first results.
- **The stacker leans on one model.** 87.5% of its split gain in version 12 is on the MS-TCN's outputs.
- **The decoder setting was not chosen again** for the five members of versions 11 and 12.
- **Training noise.** Retraining with another random seed moved the SkillCorner score by up to 0.013.
- **The match looks different** from our training games. Its ball acceleration has narrower tails than any benchmark game, and in 1 to 2% of its frames a team has fewer than 3 tracked players. Version 12 predicts 271 possessions per 90 minutes on it, against 283.5 on SkillCorner.
- **Short possessions are still missed**, most often those under 2 seconds.
- **The public 5 minutes** were never used for training, tuning or selection. We scored finished versions on them out of curiosity, most recently versions 9 and 11 after version 11 was chosen, and the full write-up lists every score. Version 12 was not scored on them.

With more time we would add professionally labelled games from more leagues.

## 7. Reproduction

```sh
pip install -r requirements-submission.txt      # Python 3.14.6
python make_submission.py                       # writes evaluation/results/camp_nou_limits.csv
```

The file (sha256 `eb249aa5…5d849f`) needs only the repository's weights and no network. On 14 Nov every check wrote this same file: macOS arm64 with 1, 2 and 8 threads, a fresh clone, Linux arm64, and x86-64 under emulation. A run takes about 4.5 minutes on one thread and 6 GB.
