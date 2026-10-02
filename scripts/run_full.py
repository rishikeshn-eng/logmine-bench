"""Run Drain on a full-size LogHub-2.0 structured CSV (from Zenodo) and report throughput + GA.

    python scripts/run_full.py path/to/HDFS_full.log_structured.csv HDFS

Needs columns Content and EventTemplate. Works on the 2k sample too (smoke test).
"""
import csv, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from logmine import metrics
from logmine.hybrid import run_hybrid
from logmine.presets import PRESETS

path, name = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8", errors="replace") as f:
    rows = list(csv.DictReader(f))
lines = [r["Content"] for r in rows]
truth_t = [r["EventTemplate"] for r in rows]
tg = {t: i for i, t in enumerate(dict.fromkeys(truth_t))}
p = PRESETS[name]
t0 = time.time()
g, _, s, _ = run_hybrid(lines, masks=p["regex"], st=p["st"], depth=p["depth"])
dt = time.time() - t0
print(f"{len(lines):,} lines in {dt:.1f}s ({len(lines)/dt:,.0f} lines/s)  clusters={s['drain_clusters']}  "
      f"hard={s['hard_clusters']}  GA={metrics.grouping_accuracy(g, [tg[t] for t in truth_t]):.4f}")
