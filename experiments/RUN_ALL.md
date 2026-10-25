# Running every experiment notebook on the dslab PC

`experiments/run_all.py` executes the notebooks in `experiments/` one after another and
writes one result JSON per notebook, so the comparison tables at the end of each notebook
(and `experiments/comparison_rmse.csv`) cover every model once the batch is done.

What it does per notebook:

1. copies `experiments/<name>.ipynb` to `experiments/runs/<name>.ipynb`;
2. in that copy only, rewrites `DEBUG = ...` / `RUN_COLLISIONS = ...` if you asked it to;
3. executes the copy with `nbconvert --execute --inplace`, console output going to
   `experiments/runs/<name>.log`;
4. the notebook itself saves `experiments/results/<EXP_NAME>.json` and its checkpoints
   under `experiments/checkpoints/<EXP_NAME>/`.

The notebooks tracked in git are never modified, and `experiments/runs/` is gitignored.
A notebook whose `results/<EXP_NAME>.json` already exists is **skipped**, so the batch is
resumable: rerun the same command after a crash and it picks up where it stopped.

## One-time setup on the dslab PC

### 1. Copy the repo over ssh

From your own machine (Git Bash / WSL), with `DSLAB=user@dslab-host`:

```bash
cd ~/OneDrive/Desktop            # the folder that holds MTP/
rsync -av --info=progress2 --exclude-from=MTP/experiments/rsync-exclude.txt MTP/ "$DSLAB:~/MTP/"
```

That sends ~1.6 GB, almost all of it the cached NGSIM CSVs in `data/raw/ngsim/` — worth it,
because the PC then needs no outbound internet for the data. `rsync` resumes, so rerun the
same command if the link drops, and again later to push notebook fixes (only the changed
files go). No rsync on either side? `scp -r MTP "$DSLAB:~/MTP"` works but copies everything
from scratch every time; exclude the big generated folders by hand, and **never** copy
`authtokens.txt`. The exclude list lives in `experiments/rsync-exclude.txt`.

Then `ssh "$DSLAB"` and keep that session for everything below.

### 2. Environment

```bash
cd ~/MTP
python3 -m venv .venv && source .venv/bin/activate      # the Windows .venv is excluded above
pip install -r requirements.txt                          # includes nbconvert + ipykernel
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True` — otherwise install the CUDA build of torch
for the box's driver; on CPU the full runs take days instead of hours.

### 3. Check the data arrived

```bash
ls -lh data/raw/ngsim/           # ngsim_us-101.csv, ngsim_i-80.csv + their .meta.json
python -m src.pipeline --mode prepare
```

`--mode prepare` builds the scenes/splits from the cache once (and downloads from the
Socrata API if a CSV is missing), so a data problem shows up now instead of failing
notebook 1 of 39 an hour into the batch.

## Running

```bash
# the real thing: all notebooks, full us-101 + i-80, detached, survives logout
mkdir -p experiments/runs
nohup python experiments/run_all.py --full > experiments/runs/run_all.out 2>&1 &
tail -f experiments/runs/run_all.out
```

Prefer `tmux new -s mtp` (then `Ctrl-b d` to detach, `tmux attach -t mtp` to come back) if
tmux is available — you get an attachable terminal instead of a log file.

Smoke-test the whole pipeline first; ~40 quick runs tell you which notebooks are broken
before you spend a night of GPU time:

```bash
python experiments/run_all.py --debug --no-collisions --force
rm -rf experiments/results experiments/checkpoints      # debug results are not comparable to full ones
```

### Options

| Flag | Effect |
|---|---|
| *(none)* | all notebooks except `_template`, each as written (they ship with `DEBUG = True`) |
| `<names>` | run only these, in the order given: `run_all.py strap_r lstm cs_lstm` (`.ipynb` optional) |
| `--full` | force `DEBUG = False`: full us-101 + i-80 — this is the real run |
| `--debug` | force `DEBUG = True`: first 250k rows of us-101, a few minutes per notebook |
| `--no-collisions` | force `RUN_COLLISIONS = False`; section 7 is the slow part on full data |
| `--force` | rerun notebooks that already have a result JSON |
| `--stop-on-fail` | stop at the first failure instead of carrying on |
| `--timeout N` | per-cell timeout in seconds (default `-1`, no limit) |
| `--list` | print the queue (after skipping finished ones) and exit |
| `--self-test` | check the DEBUG/EXP_NAME patching helpers, run nothing |

Pick a GPU on a shared box: `CUDA_VISIBLE_DEVICES=1 python experiments/run_all.py --full`.

### Output

```
[7/39] cs_lstm.ipynb ... ok (23.4 min, log: experiments/runs/cs_lstm.log)
[8/39] hltp.ipynb ... FAILED (1.2 min, log: experiments/runs/hltp.log)
    <last 15 log lines, i.e. the traceback>
...
notebook                           status       minutes
cs_lstm.ipynb                      ok             23.4
hltp.ipynb                         FAILED          1.2
```

Exit code 0 only if every notebook ran; 1 if any failed or was interrupted.

## After the batch

```bash
python -c "from experiments.benchmark import compare; print(compare(plot=False)[0])"
```

That also refreshes `experiments/comparison_rmse.csv` and `comparison_collisions.csv`.
The `data` column says `debug us-101` or `us-101,i-80`: **only rows with the same value are
comparable**, so don't mix a smoke run and a full run in one results directory.

Pull the results back over ssh, from your own machine:

```bash
rsync -av "$DSLAB:~/MTP/experiments/results/" MTP/experiments/results/   # the JSONs: all you need
rsync -av "$DSLAB:~/MTP/experiments/runs/" MTP/experiments/runs/         # optional: executed notebooks + logs
```

The JSONs are small and tracked in git, and `compare()` rebuilds both CSVs from them (the
CSVs themselves are gitignored by the `*.csv` rule), so the first command is enough to write
up the comparison locally. The executed notebooks in `runs/` carry the plots and cell output
if you want a notebook as it ran. Checkpoints are large — leave them on the PC unless you
need to re-evaluate.

## Fixing a failure

1. `less experiments/runs/<name>.log` — the traceback is at the end.
2. Reproduce the one notebook: `python experiments/run_all.py <name> --debug --force`.
3. Fix `experiments/<name>.ipynb` (the original, not the copy in `runs/`), then rerun just
   that one with `--full --force`. The rest of the results are untouched.

Common ones: CUDA OOM → lower `batch_size` in that notebook's `HPARAMS`; a notebook killed
by the OS → check `free -h` / `nvidia-smi` for another user's job on the same GPU.
