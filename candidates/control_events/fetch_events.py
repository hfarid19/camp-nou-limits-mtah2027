#!/usr/bin/env python3
"""Download the DFL event XMLs of the 7 open IDSSE games, md5-checked.

    python -I candidates/control_events/fetch_events.py [--out DIR]

Source: Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data 12:195; figshare article 28196177 version 1
(doi 10.6084/m9.figshare.28196177.v1), data by DFL Deutsche Fussball Liga, licence CC BY 4.0.

Only the 7 files DFL_03_02_events_raw_*.xml (0.6-0.8 MB each) are fetched, from
https://ndownloader.figshare.com/files/<id>. Ids, names, sizes and md5s are pinned below (copied
from the figshare listing of version 1, the same listing learned_dfl_evidence/idsse_manifest.json
pins). A file whose md5 does not match is deleted and the script fails. Files already present
with the right md5 are skipped. The files are data only and are parsed with regexes, never
executed. Default output: $MTAH_WORK/idsse/events (MTAH_WORK defaults to <repo>/external/work).
"""

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("MTAH_WORK", os.path.join(ROOT, "external", "work"))
DL = "https://ndownloader.figshare.com/files/{}"

FILES = [  # (figshare file id, name, size bytes, md5)
    (51643493, "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml", 776578, "2d88e180a1f63c7e4157feb980fc7255"),
    (51643496, "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml", 612114, "83e8ed7ea8e8ebbbdc61f61ac3534008"),
    (51643499, "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml", 635747, "ab66e51512f39d06f49f27dd3b4c2f40"),
    (51643502, "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml", 683018, "64b1be01d6ff6848168efebd56f344b9"),
    (51643505, "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml", 698898, "f68482441d38723d69f244f896482803"),
    (51643508, "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml", 732186, "b7e9d82af4034539d802319be6c57776"),
    (51643511, "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml", 681050, "c13bc5efa5eca4a022cc678210b87d5f"),
]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(fid, name, size, want, out):
    path = os.path.join(out, name)
    if os.path.isfile(path) and md5(path) == want:
        print("ok (cached)", name)
        return
    tmp = path + ".part"
    try:
        subprocess.run(["curl", "-sSfL", DL.format(fid), "-o", tmp], check=True)
    except (OSError, subprocess.CalledProcessError):
        urllib.request.urlretrieve(DL.format(fid), tmp)
    got = md5(tmp)
    if got != want or os.path.getsize(tmp) != size:
        os.remove(tmp)
        raise RuntimeError("md5/size mismatch for {}: {} != {}".format(name, got, want))
    os.replace(tmp, path)
    print("ok", name, size, got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK, "idsse", "events"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in FILES:
        fetch(*f, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
