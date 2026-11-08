#!/usr/bin/env python3
"""Download + parse the open DFL / IDSSE tracking data (Bassek et al., Sci Data 2025, CC BY 4.0).

    python -I candidates/learned_dfl_evidence/fetch_idsse.py [--games 4] [--cache DIR]

Source: figshare article 28196177 ("An integrated dataset of spatiotemporal and event data in
elite soccer", 7 Bundesliga matches, DFL Deutsche Fussball Liga / IDSSE). Files are fetched from
https://ndownloader.figshare.com/files/<id> into <cache>/raw/ (untrusted data, never executed)
and parsed line by line with regexes (no XML DOM) into <cache>/npz/<match>.npz with

    players (N, P, 2) float32 metres in 0..105 x 0..68 (NaN when the person is not on the pitch)
    team_of_player (P,) 0 home, 1 away
    ball (N, 2) float32, poss (N,) int8 (1 home, 2 away, 0 unknown), status (N,) int8 (0 dead, 1 alive)
    half (N,) int8, frame (N,) int32 (DFL frame number)

Rows are the union of frames of each half, halves concatenated.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# raw data + caches: $IDSSE_CACHE, else <candidate>/external/idsse (git-ignored); --cache overrides
DEFAULT_CACHE = os.environ.get("IDSSE_CACHE", os.path.join(HERE, "external", "idsse"))
ARTICLE = "https://api.figshare.com/v2/articles/28196177/versions/1"   # pinned dataset version
MANIFEST = os.path.join(HERE, "idsse_manifest.json")   # (id, name, size, computed_md5), committed
DL = "https://ndownloader.figshare.com/files/{}"

RE_FS = re.compile(r'<FrameSet\b([^>]*)>')
RE_ATTR = re.compile(r'(\w+)="([^"]*)"')
RE_FRAME = re.compile(r'<Frame\b([^/]*)/>')


def _manifest():
    with open(MANIFEST) as handle:
        return {f["id"]: f for f in json.load(handle)["files"]}


def file_list(cache):
    """Files of the pinned figshare version; must agree with the committed manifest."""
    path = os.path.join(cache, "article.json")
    if not os.path.isfile(path):
        os.makedirs(cache, exist_ok=True)
        urllib.request.urlretrieve(ARTICLE, path)
    with open(path) as handle:
        art = json.load(handle)
    man = _manifest()
    files = [(f["id"], f["name"], f["size"]) for f in art["files"]]
    for fid, name, size in files:
        if fid in man and (man[fid]["name"], man[fid]["size"]) != (name, size):
            raise RuntimeError("figshare listing differs from idsse_manifest.json: " + name)
    return files


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(fid, path):
    want = _manifest().get(fid, {}).get("computed_md5")
    if want and _md5(path) != want:
        os.remove(path)
        raise RuntimeError("md5 mismatch for {} (expected {})".format(path, want))
    return path


def download(fid, name, size, cache):
    raw = os.path.join(cache, "raw")
    os.makedirs(raw, exist_ok=True)
    path = os.path.join(raw, name)
    if os.path.isfile(path) and os.path.getsize(path) == size:
        return _verify(fid, path)
    tmp = path + ".part"
    print("downloading", name, size // 2 ** 20, "MB", flush=True)
    if shutil.which("curl"):                       # much faster than urllib on figshare
        subprocess.run(["curl", "-sSL", DL.format(fid), "-o", tmp], check=True)
        os.replace(tmp, path)
        return _verify(fid, path)
    with urllib.request.urlopen(DL.format(fid)) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, path)
    return _verify(fid, path)


def parse_info(path):
    """{team_id: 0 home / 1 away}, {person_id: team_id} from the matchinformation XML."""
    text = open(path, encoding="utf-8").read()
    teams = {}
    for m in re.finditer(r'<Team\b([^>]*)>', text):
        a = dict(RE_ATTR.findall(m.group(1)))
        if "TeamId" in a and "Role" in a:
            teams[a["TeamId"]] = 0 if a["Role"] == "home" else 1
    persons = {}
    for m in re.finditer(r'<Team\b([^>]*)>(.*?)</Team>', text, re.S):
        a = dict(RE_ATTR.findall(m.group(1)))
        for p in re.finditer(r'<Player\b([^>]*)/?>', m.group(2)):
            pa = dict(RE_ATTR.findall(p.group(1)))
            if "PersonId" in pa:
                persons[pa["PersonId"]] = a.get("TeamId")
    return teams, persons


def parse_positions(path, teams, verbose=True):
    sets = {}      # (section, person) -> dict(team, frames list, x, y, extra attrs)
    cur = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if "<Frame " in line:
                if cur is None:
                    continue
                a = RE_FRAME.search(line)
                if not a:
                    continue
                d = dict(RE_ATTR.findall(a.group(1)))
                cur["N"].append(int(d["N"]))
                cur["X"].append(float(d["X"]))
                cur["Y"].append(float(d["Y"]))
                if cur["ball"]:
                    cur["poss"].append(int(d.get("BallPossession", 0) or 0))
                    cur["status"].append(int(d.get("BallStatus", -1) or -1))
                    if cur["first"] is None:
                        cur["first"] = d
            elif "<FrameSet" in line:
                m = RE_FS.search(line)
                d = dict(RE_ATTR.findall(m.group(1)))
                is_ball = d.get("TeamId", "").upper() == "BALL"
                key = (d["GameSection"], d["PersonId"])
                cur = sets.setdefault(key, dict(team=d.get("TeamId"), ball=is_ball, N=[], X=[], Y=[],
                                                poss=[], status=[], first=None))
            elif "</FrameSet>" in line:
                cur = None
    sections = sorted({k[0] for k in sets}, key=lambda s: 0 if s == "firstHalf" else 1)
    persons = sorted({k[1] for k, v in sets.items() if not v["ball"] and v["team"] in teams})
    pidx = {p: i for i, p in enumerate(persons)}
    team_of = np.zeros(len(persons), np.int8)
    for (sec, p), v in sets.items():
        if p in pidx:
            team_of[pidx[p]] = teams[v["team"]]
    out = dict(players=[], ball=[], poss=[], status=[], half=[], frame=[])
    for h, sec in enumerate(sections):
        ballset = [v for (s, _), v in sets.items() if s == sec and v["ball"]]
        assert len(ballset) == 1, (sec, len(ballset))
        b = ballset[0]
        if verbose and h == 0:
            print("  ball FrameSet first frame attrs:", b["first"])
        allN = np.concatenate([np.asarray(v["N"]) for (s, _), v in sets.items() if s == sec])
        n0, n1 = int(allN.min()), int(allN.max())
        n = n1 - n0 + 1
        P = np.full((n, len(persons), 2), np.nan, np.float32)
        for (s, p), v in sets.items():
            if s != sec or p not in pidx:
                continue
            idx = np.asarray(v["N"]) - n0
            P[idx, pidx[p], 0] = np.asarray(v["X"], np.float32) + 52.5
            P[idx, pidx[p], 1] = np.asarray(v["Y"], np.float32) + 34.0
        B = np.full((n, 2), np.nan, np.float32)
        po = np.zeros(n, np.int8)
        st = np.full(n, -1, np.int8)
        idx = np.asarray(b["N"]) - n0
        B[idx, 0] = np.asarray(b["X"], np.float32) + 52.5
        B[idx, 1] = np.asarray(b["Y"], np.float32) + 34.0
        po[idx] = b["poss"]
        st[idx] = b["status"]
        out["players"].append(P)
        out["ball"].append(B)
        out["poss"].append(po)
        out["status"].append(st)
        out["half"].append(np.full(n, h + 1, np.int8))
        out["frame"].append(np.arange(n0, n1 + 1, dtype=np.int32))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["team_of_player"] = team_of
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--matches", default="", help="comma-separated match ids (default: first --games)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.cache, "npz"), exist_ok=True)
    files = file_list(args.cache)
    infos = {re.search(r'(DFL-MAT-\w+)', n).group(1): (i, n, s) for i, n, s in files
             if "matchinformation" in n}
    poss = sorted([(re.search(r'(DFL-MAT-\w+)', n).group(1), (i, n, s)) for i, n, s in files
                   if "positions" in n])
    todo = poss[:args.games]
    if args.matches:
        todo = [p for p in poss if p[0] in args.matches.split(",")]
    for match, (fid, name, size) in todo:
        out = os.path.join(args.cache, "npz", match + ".npz")
        if os.path.isfile(out):
            print("cached", out)
            continue
        info = download(*infos[match], args.cache)
        teams, _ = parse_info(info)
        raw = download(fid, name, size, args.cache)
        print("parsing", name, flush=True)
        res = parse_positions(raw, teams)
        st, po = res["status"], res["poss"]
        print("  frames", len(st), "persons", res["players"].shape[1],
              "status counts", dict(zip(*np.unique(st, return_counts=True))),
              "poss counts", dict(zip(*np.unique(po, return_counts=True))), flush=True)
        np.savez_compressed(out, **res)
        if not args.keep_raw:
            os.remove(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
