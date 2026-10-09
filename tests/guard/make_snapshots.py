"""Generate the bitwise snapshots of every libspatialize function on this machine.

Run ONLY on the code BEFORE a refactor (snapshots describe that code, bugs included):

    make && DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python tests/guard/make_snapshots.py

(the DYLD_LIBRARY_PATH is needed on macOS with a conda Python, see the README).
"""
import argparse
import datetime
import subprocess
import time

import cases
import public_cases
import snapshot_lib as sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="case names to (re)generate")
    args = ap.parse_args()

    lib = sl.load_lib()
    data = cases.datasets()
    commit = subprocess.run(["git", "-C", sl.REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", sl.REPO, "status", "--porcelain", "--", "src/c++", "include/spatialize"],
                                capture_output=True, text=True).stdout.strip())
    for name, ds, fn in cases.cases(lib) + public_cases.public_cases():
        if args.only and name not in args.only:
            continue
        s, v, q = data[ds]
        t0 = time.perf_counter()
        out = fn(s, v, q) if name.startswith("api.") else cases.normalise(fn(s, v, q))
        meta = {"case": name, "dataset": ds, "platform": sl.platform_tag(), "commit": commit,
                "cpp_dirty": dirty, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "seconds": round(time.perf_counter() - t0, 3)}
        sl.save(name, {"samples": s, "values": v, "queries": q}, out, meta)
        print(f"{name:40s} {meta['seconds']:7.2f}s  " + ", ".join(f"{k}{list(a.shape)}" for k, a in out.items())[:120])


if __name__ == "__main__":
    main()
