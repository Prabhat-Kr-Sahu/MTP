#!/usr/bin/env python3
"""Run the experiment notebooks in experiments/ one after another.

Each notebook is copied to experiments/runs/<file>.ipynb, patched there if asked
(DEBUG / RUN_COLLISIONS), and executed with nbconvert, so the notebooks tracked in
git keep their committed outputs and the executed copy keeps the run's output next
to its log in experiments/runs/. Results still land in experiments/results/<EXP_NAME>.json,
which is also what makes a notebook count as already done (skipped unless --force).

  python experiments/run_all.py --full            # all notebooks, full NGSIM data
  python experiments/run_all.py lstm cs_lstm      # just these two, as written
See experiments/RUN_ALL.md.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
RESULTS = EXPERIMENTS / "results"
RUNS = EXPERIMENTS / "runs"
SKIP = {"_template"}            # not a real experiment


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exp_name(nb, default):
    """EXP_NAME from the notebook's config cell -> results/<EXP_NAME>.json."""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            m = re.search(r'^EXP_NAME\s*=\s*"([^"]+)"', "".join(cell["source"]), re.M)
            if m:
                return m.group(1)
    return default


def patch(nb, debug=None, collisions=None):
    """Rewrite the `DEBUG = ...` / `RUN_COLLISIONS = ...` assignments, trailing comments kept."""
    subs = [("DEBUG", debug), ("RUN_COLLISIONS", collisions)]
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for var, val in subs:
            if val is not None:
                src = re.sub(rf"^{var} = (?:True|False)", f"{var} = {val}", src, flags=re.M)
        cell["source"] = src.splitlines(keepends=True)
    return nb


def run_one(src_nb, args):
    """Execute one notebook copy; returns (status, minutes, log_path)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    out_nb = RUNS / src_nb.name
    log = RUNS / f"{src_nb.stem}.log"
    out_nb.write_text(json.dumps(patch(load(src_nb), args.debug, args.collisions), indent=1),
                      encoding="utf-8")

    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           f"--ExecutePreprocessor.timeout={args.timeout}",
           "--ExecutePreprocessor.kernel_name=python3", str(out_nb)]
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\nstarted {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        fh.flush()
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=EXPERIMENTS)
    return ("ok" if code == 0 else "FAILED"), (time.time() - t0) / 60, log


def tail(path, n=15):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(f"    {line}" for line in lines[-n:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebooks", nargs="*", help="notebook names/paths; default = all of them")
    data = p.add_mutually_exclusive_group()
    data.add_argument("--full", dest="debug", action="store_const", const=False,
                      help="force DEBUG = False (full us-101 + i-80)")
    data.add_argument("--debug", dest="debug", action="store_const", const=True,
                      help="force DEBUG = True (250k rows of us-101, quick smoke run)")
    p.set_defaults(debug=None)
    p.add_argument("--no-collisions", dest="collisions", action="store_const", const=False,
                   default=None, help="force RUN_COLLISIONS = False (the slow part on full data)")
    p.add_argument("--force", action="store_true", help="rerun even if results/<name>.json exists")
    p.add_argument("--stop-on-fail", action="store_true", help="stop at the first failure")
    p.add_argument("--timeout", type=int, default=-1, help="per-cell timeout in s (-1 = none)")
    p.add_argument("--list", action="store_true", help="print what would run, then exit")
    p.add_argument("--self-test", action="store_true", help="check the patching helpers")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.notebooks:
        paths = [EXPERIMENTS / f"{Path(n).stem}.ipynb" for n in args.notebooks]
        missing = [str(q) for q in paths if not q.exists()]
        if missing:
            p.error("no such notebook: " + ", ".join(missing))
    else:
        paths = [q for q in sorted(EXPERIMENTS.glob("*.ipynb")) if q.stem not in SKIP]

    queue = []
    for path in paths:
        name = exp_name(load(path), path.stem)
        done = (RESULTS / f"{name}.json").exists()
        if done and not args.force:
            print(f"skip {path.name}: results/{name}.json exists (--force to rerun)")
            continue
        queue.append(path)

    if args.list or not queue:
        print(f"{len(queue)} notebook(s) to run:" if queue else "nothing to run.")
        for path in queue:
            print(" ", path.name)
        return 0

    results, t0 = [], time.time()
    for i, path in enumerate(queue, 1):
        print(f"[{i}/{len(queue)}] {path.name} ... ", end="", flush=True)
        try:
            status, minutes, log = run_one(path, args)
        except KeyboardInterrupt:
            print("interrupted")
            results.append((path.name, "interrupted", 0.0))
            break
        print(f"{status} ({minutes:.1f} min, log: {log.relative_to(EXPERIMENTS.parent)})")
        if status != "ok":
            print(tail(log))
        results.append((path.name, status, minutes))
        if status != "ok" and args.stop_on_fail:
            break

    print(f"\n{'notebook':34} {'status':12} minutes")
    for name, status, minutes in results:
        print(f"{name:34} {status:12} {minutes:6.1f}")
    print(f"total {(time.time() - t0) / 60:.1f} min; "
          f"compare with: python -c \"from experiments.benchmark import compare; "
          f"print(compare(plot=False)[0])\"")
    return 0 if all(s == "ok" for _, s, _ in results) else 1


def self_test():
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["set `DEBUG = False` for the real run\n"]},
        {"cell_type": "code", "source": ['EXP_NAME = "lstm"   # unique\n',
                                         "DEBUG = True        # note\n",
                                         "RUN_COLLISIONS = True\n"]},
    ]}
    assert exp_name(nb, "stem") == "lstm"
    assert exp_name({"cells": []}, "stem") == "stem"
    code = "".join(patch(nb, debug=False, collisions=False)["cells"][1]["source"])
    assert "DEBUG = False        # note" in code, code      # comment kept
    assert "RUN_COLLISIONS = False" in code, code
    assert "`DEBUG = False`" in nb["cells"][0]["source"][0]  # markdown untouched
    nb2 = {"cells": [{"cell_type": "code", "source": ["DEBUG = False\n"]}]}
    assert "DEBUG = True" in "".join(patch(nb2, debug=True)["cells"][0]["source"])
    assert "DEBUG = True" in "".join(patch(nb2, debug=None)["cells"][0]["source"])  # no-op
    print("self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
