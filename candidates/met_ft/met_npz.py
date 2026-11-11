#!/usr/bin/env python3
"""Metrica Sample Games 1-3 in the game-npz format of learned_dfl_evidence/make_dataset.load_game (the DFL / PFF
training npz: players, ball, team_of_player, poss, status, half, frame), with Metrica-rule strict labels, for the
radar (ball_radar_pff) retrain of round-5 prereg addendum 2026-11-11 12:50 CST (opt-in BRP_MET=1 in brp_common).

    python -I candidates/met_ft/met_npz.py [--out $MTAH_WORK/met_npz]

Tracking exactly as the Metrica bench loads it (tune_on_sample_data loaders: metres, 105 x 68, corner origin, 25 fps).
Labels: control_events/metrica_rules.strict_labels over tsd.practice_labels (as met_data.py), encoded so that
make_dataset.labels_blocks returns them: H -> status 1 / poss 1, A -> status 1 / poss 2, D -> status 0, X -> status 1 /
poss 0; half = h inside tsd's block h, else 0. Checks (asserted / printed): labels_blocks codes == strict labels on every
frame; its blocks vs tsd's; the radar's kick-off sign rule (prep_data._kickoff_sign) == tsd.attack_sign.
Reads only Metrica's open sample data.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), CAND, os.path.join(CAND, "control_events"),
                os.path.join(CAND, "learned_dfl_evidence")]
import tune_on_sample_data as tsd  # noqa: E402
import metrica_rules as mr         # noqa: E402
import make_dataset as md          # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
SAMPLE = os.path.join(WORK, "sample", "sample-data-master", "data")
GAMES = {"MET-G1": "Sample_Game_1", "MET-G2": "Sample_Game_2", "MET-G3": "Sample_Game_3"}


def kickoff_sign(players, team_of_player, blocks):
    """= ball_radar_pff/prep_data._kickoff_sign (home players over the first 50 frames of the first half)."""
    _, s, _ = blocks[0]
    home = players[s:s + 50, team_of_player == 0, 0]
    return 1.0 if np.nanmean(home) < md.PITCH[0] / 2 else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK, "met_npz"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, sample in GAMES.items():
        folder = os.path.join(SAMPLE, sample)
        game = tsd.load_epts_game(folder) if sample == "Sample_Game_3" else tsd.load_csv_game(folder)
        prac, blocks = tsd.practice_labels(game)
        strict = mr.strict_labels(game, prac, blocks)
        n = len(strict)
        status = np.ones(n, np.int8)
        poss = np.zeros(n, np.int8)
        poss[strict == 0] = 1
        poss[strict == 1] = 2
        status[strict == 2] = 0
        half = np.zeros(n, np.int8)
        for h, f, l in blocks:
            half[f:l + 1] = h
        out = dict(players=game.players.astype(np.float32), ball=game.ball.astype(np.float32), poss=poss, status=status,
                   half=half, frame=np.arange(n, dtype=np.int32), team_of_player=np.asarray(game.team_of_player, np.int8))
        codes, bl = md.labels_blocks(out)
        inside = half > 0
        assert np.array_equal(codes[inside], strict[inside]), name
        assert (codes[~inside] == 3).all(), name
        s_tsd = float(tsd.attack_sign(game, blocks))
        s_rad = kickoff_sign(out["players"], out["team_of_player"], bl)
        assert s_tsd == s_rad, (name, s_tsd, s_rad)
        alive = strict < 2
        print("{} ({}): frames {} players {} | strict codes H/A/D/X {} | tsd blocks {} | labels_blocks {} | "
              "strict outside blocks: {} non-X | sign {} | clean ball missing: all {:.3f}, alive {:.3f}, dead {:.3f} | "
              "x [{:.1f}, {:.1f}] y [{:.1f}, {:.1f}]".format(
                  name, sample, n, out["players"].shape[1], np.bincount(strict, minlength=4).tolist(),
                  [tuple(map(int, b)) for b in blocks], bl, int((strict[~inside] != 3).sum()), s_tsd,
                  float(np.isnan(out["ball"][:, 0]).mean()), float(np.isnan(out["ball"][alive, 0]).mean()),
                  float(np.isnan(out["ball"][strict == 2, 0]).mean()),
                  float(np.nanmin(out["players"][..., 0])), float(np.nanmax(out["players"][..., 0])),
                  float(np.nanmin(out["players"][..., 1])), float(np.nanmax(out["players"][..., 1]))), flush=True)
        p = os.path.join(a.out, name + ".npz")
        np.savez_compressed(p + ".tmp.npz", **out)
        os.replace(p + ".tmp.npz", p)


if __name__ == "__main__":
    main()
