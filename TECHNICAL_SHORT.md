# Ball Possession Prediction Using Tracking Data and Open Labelled Games

Team Camp Nou Limits, More than a Hack 2027, Phase 1. This is the short document. The full write-up is [`TECHNICAL.md`](TECHNICAL.md) in the repository.

## 1. Approach

We sought to label every frame of the match without ever labelling the provided match. Open tracking data from other games already carries possession labels. Therefore, we learned possession from those games and degraded their clean tracking until it looked like the automatic tracking of the provided match. The degradation was fitted to statistics of the match that need no labels.

![How the pipeline works](docs/fig/pipeline.png)

Four models read each game: a temporal convolutional network (TCN), a multi-stage network that also finds possession boundaries (MS-TCN), a ball radar that infers the missing ball from player movement, and a pointer that picks the possessor among the 8 players nearest the ball. A minimum Bayes risk (MBR) decoder combines them. It keeps the labelling with the highest expected spellF1, which is the competition metric.

![First and final submission](docs/fig/short_first_final.png)

## 2. Training Data

Nothing comes from the provided game's labels, and no person on our team labelled anything by hand. Every score in this document comes from models that never saw the scored game.

| Dataset | Games | Labels | Use |
|---|---|---|---|
| DFL / IDSSE Bundesliga (CC BY 4.0) | 7 | The league's official possession flags | Training, main benchmark |
| PFF FC World Cup 2022 | 64 | Our script, from PFF's events | 48 training, 16 held out |
| SkillCorner A-League (MIT) | 20 | Our script, from SkillCorner's events | Benchmark, fine-tuning one model |
| Metrica Sports sample games | 3 | Our script, from Metrica's events | Benchmark, fine-tuning one model |
| FOOTPASS / SoccerNet | 51 | Our script, from its events | 13-game holdout, scored once, local only |

## 3. Assets Used and Dropped

We used the tactical tracking as the main input and the Smart Tagging tags for small priors, such as dead ball before each kick-off. We did not use the video or the homography, since the tracking is already in pitch coordinates.

![Ideas we tested and dropped](docs/fig/short_dropped.png)

The 11M ball radar was clearly better alone and added nothing to the ensemble, because the other three models already covered what it knew.

## 4. Possession Rules

- **Dead ball** wherever the models believe the ball is out of play, and before each kick-off.
- **Loose balls and passes in flight** stay with the last team in possession.
- **Controlled possession.** An opponent possession shorter than 1.25 seconds is handed back, since a touch without control does not change possession.
- **After a stoppage** the first possession is kept even if it is short.

The Bundesliga flag changes on every touch, so we needed to know how finely the truth counts possessions. Metrica's and PFF's counts sit between a 1 second and a 1.5 second threshold. Our first submission assumed 2 seconds, and correcting this was our largest single gain.

![Possession counts by provider](docs/fig/short_counts.png)

## 5. Results

**The decoder.** On the same four models, MBR decoding beat a plain vote on every Bundesliga game.

![Decoder against vote](docs/fig/short_decoder.png)

**The final step.** Metrica co-organises the challenge and produced the match's tracking and tagging, so we expect the hidden truth to follow Metrica's way of labelling. We fine-tuned the MS-TCN on Metrica's three sample games and accepted a small cost on the other benchmarks.

![The Metrica trade-off](docs/fig/short_tradeoff.png)

**Model size.** We trained every model family at sizes up to 500 million parameters. This matches published scaling laws: with a fixed dataset, loss stops improving as models grow (Kaplan et al., 2020), and repeated training data makes it worse before it gets better (Hernandez et al., 2022). With about 50 labelled games, data is our limit.

![Score against model size](docs/fig/short_size.png)

We also saw double descent. In a transformer family that we tested and did not ship (ASFormer), held-out loss got worse as the model grew to 32 million parameters and then fell below every smaller model at 198 and 505 million. The sizes up to 32 million also changed depth, so the fall above 32 million is the clean part of this result. It did not carry over to SkillCorner.

![Double descent in held-out loss](docs/fig/short_dd.png)

## 6. Limitations

- **The benchmarks are proxies.** If the truth counts possessions more coarsely than we believe, our output would cost about 0.04.
- **We reused small benchmarks** for every choice, so the scores above are optimistic. The final version has no once-only confirmation. An earlier one gained +0.030 on 13 fresh games.
- **The final step's rule was changed after its first results**, to give Metrica more weight. Three Metrica games is thin evidence.
- **Training noise.** Retraining with another random seed moved the SkillCorner score by up to 0.013.
- **The match looks different** from our training games, and the models predict fewer possessions on it (273 per 90 minutes) than on any benchmark (285 to 309).
- **Short possessions are still missed**, most often those under 2 seconds.
- **The public 5 minutes** were never used for training, tuning or selection. We scored finished versions on them out of curiosity, and the full write-up lists every score.

With more time we would add professionally labelled games from more leagues.

## 7. Reproduction

```sh
pip install -r requirements-submission.txt      # Python 3.14.6
python make_submission.py                       # writes evaluation/results/camp_nou_limits.csv
```

The file is regenerated byte for byte (sha256 `9926b7c5…bcb3e2`) from committed weights, with no network access. We obtained the identical file on macOS with 1, 2 and 8 threads, in Linux arm64 containers including a fresh clone, and on Linux x86-64 under emulation. It takes 2 to 20 minutes.
