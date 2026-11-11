"""Weights files split into chunks (GitHub refuses files over 100 MB), and loaders that read either form.

    python candidates/chunked_weights.py split <file.npz> [max_bytes]    # writes the parts + <file.npz>.sha256
    python candidates/chunked_weights.py check <file.npz>                # re-joins the parts, checks the sha256

split(path, max_bytes=95_000_000) writes <path>.part000, <path>.part001, ... (consecutive byte slices of the file,
each at most max_bytes, nothing added) and <path>.sha256 (one line: the sha256 of the whole file, two spaces, its
base name). It leaves the original file in place; delete it (or keep it out of git) yourself.

When <path> exists it is always used as it is: source(path) returns the path, open_weights(path) is np.load(path) and
sha256(path) is the sha256 of the file. Only when <path> is missing are the parts read: they are concatenated in
order (part000, part001, ... up to the first missing number) in memory, the sha256 of the result is checked against
<path>.sha256, and the bytes are served from memory. Model resolution that keys on a weights file's sha256
(dfl_folds, pff_folds) uses sha256(path), so it sees the same value for a file and for its re-joined parts.
numpy + standard library only; deterministic.
"""

import hashlib
import io
import os
import sys

import numpy as np

MAX_BYTES = 95_000_000
_JOINED = {}            # path -> verified bytes (the parts are read once per process)


def part_name(path, i):
    return "{}.part{:03d}".format(path, i)


def parts(path):
    """The existing part files of path, in order (part000, part001, ... up to the first gap)."""
    out = []
    while os.path.isfile(part_name(path, len(out))):
        out.append(part_name(path, len(out)))
    return out


def split(path, max_bytes=MAX_BYTES):
    """Write path's parts and its .sha256 file; returns the list of part paths. Stale higher-numbered parts from an
    earlier split are removed so that the part list is exactly this file."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    with open(path, "rb") as handle:
        data = handle.read()
    out = []
    for i, start in enumerate(range(0, max(len(data), 1), max_bytes)):
        out.append(part_name(path, i))
        with open(out[-1], "wb") as handle:
            handle.write(data[start:start + max_bytes])
    i = len(out)
    while os.path.isfile(part_name(path, i)):
        os.remove(part_name(path, i))
        i += 1
    with open(path + ".sha256", "w") as handle:
        handle.write("{}  {}\n".format(hashlib.sha256(data).hexdigest(), os.path.basename(path)))
    return out


def _joined(path):
    key = os.path.abspath(path)
    if key not in _JOINED:
        files = parts(path)
        if not files:
            raise FileNotFoundError("{}: neither the file nor {} exists".format(path, part_name(path, 0)))
        sha_file = path + ".sha256"
        if not os.path.isfile(sha_file):
            raise FileNotFoundError("{}: parts found but no {}".format(path, sha_file))
        with open(sha_file) as handle:
            want = handle.read().split()[0].lower()
        chunks = []
        for f in files:
            with open(f, "rb") as handle:
                chunks.append(handle.read())
        data = b"".join(chunks)
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise ValueError("{}: the {} parts join to sha256 {}, but {} says {}".format(
                path, len(files), got, sha_file, want))
        _JOINED[key] = data
    return _JOINED[key]


def exists(path):
    return os.path.isfile(path) or os.path.isfile(part_name(path, 0))


def source(path):
    """What np.load should read: path itself if the file exists, else the verified re-joined parts (in memory)."""
    if os.path.isfile(path):
        return path
    return io.BytesIO(_joined(path))


def open_weights(path):
    """np.load(path) for a single file, or the same NpzFile read from the re-joined parts."""
    return np.load(source(path))


def read_bytes(path):
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return handle.read()
    return _joined(path)


def sha256(path):
    """sha256 (hex) of the file's content, or of the re-joined parts when the file is missing."""
    return hashlib.sha256(read_bytes(path)).hexdigest()


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "split":
        files = split(path, int(sys.argv[3]) if len(sys.argv) > 3 else MAX_BYTES)
        print("{} -> {} parts, sha256 {}".format(path, len(files), open(path + ".sha256").read().split()[0]))
    elif cmd == "check":
        data = _joined(path)
        print("{}: {} parts, {} bytes, sha256 ok".format(path, len(parts(path)), len(data)))
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                print("equal to the single file:", handle.read() == data)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
