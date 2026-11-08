#!/usr/bin/env python3
"""Download the SkillCorner open data (github.com/SkillCorner/opendata, MIT licence) at a pinned
commit and check every tracking file against the sha256 / size in its git-LFS pointer.

    python -I fetch_skillcorner.py --out DIR [--matches N]

Writes DIR/<id>/<id>_{match.json, tracking_extrapolated.jsonl, phases_of_play.csv,
dynamic_events.csv} and DIR/matches.json. Nothing is committed; DIR lives outside the repo.

Source: SkillCorner Open Data, 20 A-League 2024/25 matches of broadcast tracking (10 fps) with
a per-frame `possession` {player_id, group} field, Dynamic Events and Phases of Play.
Licence: MIT (Copyright (c) 2020 SkillCorner), see the repository's LICENSE. SkillCorner asks
users to credit them.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

COMMIT = "4340d274572876239c154c90bc507a9b3250a656"   # master on 2026-11-07
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/".format(COMMIT)
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/".format(COMMIT)
TREE = "https://api.github.com/repos/SkillCorner/opendata/git/trees/{}?recursive=1".format(COMMIT)


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as exc:          # transient network errors
            if k == tries - 1:
                raise
            print("retry", url, exc, file=sys.stderr)
            time.sleep(3 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--matches", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tree = json.loads(get(TREE))["tree"]
    ids = sorted({p["path"].split("/")[2] for p in tree
                  if p["path"].startswith("data/matches/") and p["path"].count("/") == 3})
    if args.matches:
        ids = ids[:args.matches]
    with open(os.path.join(args.out, "matches.json"), "wb") as h:
        h.write(get(RAW + "data/matches.json"))
    manifest = {}
    for mid in ids:
        d = os.path.join(args.out, mid)
        os.makedirs(d, exist_ok=True)
        for suffix in ("match.json", "phases_of_play.csv", "dynamic_events.csv"):
            path = os.path.join(d, "{}_{}".format(mid, suffix))
            if not os.path.isfile(path):
                with open(path, "wb") as h:
                    h.write(get(RAW + "data/matches/{}/{}_{}".format(mid, mid, suffix)))
            with open(path, "rb") as h:
                manifest["{}_{}".format(mid, suffix)] = hashlib.sha256(h.read()).hexdigest()
        rel = "data/matches/{}/{}_tracking_extrapolated.jsonl".format(mid, mid)
        pointer = get(RAW + rel).decode()
        oid = pointer.split("oid sha256:")[1].split()[0]
        size = int(pointer.split("size ")[1].split()[0])
        path = os.path.join(d, "{}_tracking_extrapolated.jsonl".format(mid))
        ok = False
        if os.path.isfile(path) and os.path.getsize(path) == size:
            with open(path, "rb") as h:
                ok = hashlib.sha256(h.read()).hexdigest() == oid
        if not ok:
            t0 = time.time()
            blob = get(MEDIA + rel)
            assert len(blob) == size and hashlib.sha256(blob).hexdigest() == oid, rel
            with open(path, "wb") as h:
                h.write(blob)
            print("{} {:.0f} MB {:.0f}s".format(mid, size / 1e6, time.time() - t0), flush=True)
        manifest["{}_tracking_extrapolated.jsonl".format(mid)] = oid
    with open(os.path.join(args.out, "manifest.json"), "w") as h:
        json.dump(dict(commit=COMMIT, sha256=manifest), h, indent=1)
    print("ok", len(ids), "matches")


if __name__ == "__main__":
    main()
