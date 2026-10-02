"""Benchmark runner: Drain vs hybrid on LogHub-2k, with published Drain baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data, metrics
from .hybrid import run_hybrid
from .llm import OracleLLM, GeminiLLM, cost_usd
from .spikes import window_counts, spikes
from .presets import PRESETS

# Published Drain results on Loghub_2k, copied from
# https://github.com/logpai/logparser/blob/main/logparser/Drain/README.md ("Accuracy" = grouping accuracy).
PUBLISHED_DRAIN_GA = {
    "HDFS": 0.9975, "Hadoop": 0.9475, "Spark": 0.92, "Zookeeper": 0.9665, "BGL": 0.9625,
    "HPC": 0.887, "Thunderbird": 0.955, "Windows": 0.997, "Linux": 0.69, "Android": 0.911,
    "HealthApp": 0.78, "Apache": 1.0, "Proxifier": 0.5265, "OpenSSH": 0.7875,
    "OpenStack": 0.7325, "Mac": 0.7865,
}


def tree_payload(d, truth_groups, per_line_templates, windows=40, top=60):
    n_cl = len(d.clusters)
    cids = [None] * sum(c.size for c in d.clusters)
    for c in d.clusters:
        for i in c.line_ids:
            cids[i] = c.cid
    wc = window_counts(cids, n_cl, windows)
    out = []
    for c in sorted(d.clusters, key=lambda c: -c.size)[:top]:
        out.append({"template": c.text(), "count": c.size, "series": wc[c.cid].tolist(),
                    "spikes": spikes(wc[c.cid]), "example": c.examples[0][:160]})
    return out


def run(names, llm_mode: str, price_in: float, price_out: float, out_dir: Path, cache: Path,
        tuned: bool = True):
    rows_out, trees = [], {}
    for name in names:
        rows = data.load(name, cache)
        lines = [r["Content"] for r in rows]
        truth_t = [r["EventTemplate"] for r in rows]
        tg = {t: i for i, t in enumerate(dict.fromkeys(truth_t))}
        truth = [tg[t] for t in truth_t]

        kw = ({"masks": PRESETS[name]["regex"], "st": PRESETS[name]["st"], "depth": PRESETS[name]["depth"]}
              if tuned else {})
        g0, t0, s0, d0 = run_hybrid(lines, **kw)
        gg, _, _, _ = run_hybrid(lines)  # generic masks, default params
        row = {"dataset": name, "lines": len(lines), "true_templates": len(tg),
               "drain_GA": metrics.grouping_accuracy(g0, truth),
               "drain_F1": metrics.pairwise_f1(g0, truth),
               "generic_GA": metrics.grouping_accuracy(gg, truth),
               "published_drain_GA": PUBLISHED_DRAIN_GA[name],
               "drain_clusters": s0["drain_clusters"], "hard_clusters": s0["hard_clusters"],
               "hard_line_share": s0["hard_lines"] / len(lines)}
        if llm_mode != "none":
            llm = OracleLLM() if llm_mode == "oracle" else GeminiLLM()
            g1, t1, s1, _ = run_hybrid(lines, llm, truth_templates=truth_t, **kw)
            row.update({"hybrid_GA": metrics.grouping_accuracy(g1, truth),
                        "hybrid_F1": metrics.pairwise_f1(g1, truth),
                        "llm_calls": llm.usage.calls, "llm_rejected": llm.usage.rejected,
                        "llm_tokens_in": llm.usage.tokens_in, "llm_tokens_out": llm.usage.tokens_out,
                        "usd_per_million_lines_upper": cost_usd(llm.usage, price_in, price_out)
                        / len(lines) * 1e6})
        rows_out.append(row)
        trees[name] = tree_payload(d0, truth, t0)
        print(f"{name:12s} generic={row['generic_GA']:.3f} GA={row['drain_GA']:.3f} (published {row['published_drain_GA']:.3f}) "
              f"hard_clusters={row['hard_clusters']}"
              + (f" hybrid_GA={row['hybrid_GA']:.3f}" if "hybrid_GA" in row else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"llm_mode": llm_mode, "price_in_per_m": price_in, "price_out_per_m": price_out,
               "results": rows_out, "trees": trees}
    (out_dir / "results.json").write_text(json.dumps(payload))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=data.DATASETS)
    ap.add_argument("--llm", choices=["none", "oracle", "gemini"], default="oracle")
    ap.add_argument("--price-in", type=float, default=0.10, help="USD per 1M input tokens (PLACEHOLDER: set from current pricing)")
    ap.add_argument("--price-out", type=float, default=0.40, help="USD per 1M output tokens (PLACEHOLDER)")
    ap.add_argument("--generic", action="store_true", help="use generic masks/params instead of per-dataset presets")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    run(a.datasets, a.llm, a.price_in, a.price_out, Path(a.out), Path("data"), tuned=not a.generic)


if __name__ == "__main__":
    main()
