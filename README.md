# logmine-bench

Log template mining with a cost-aware LLM escalation path, benchmarked on real [LogHub](https://github.com/logpai/loghub) logs.

**Live UI:** https://rishikeshn-eng.github.io/logmine-bench/ (benchmark table + template tree with anomaly spikes)

## What it does

1. **Drain** (own implementation, `logmine/drain.py`) parses every line into templates.
2. Clusters that look wrong (tiny support = probably over-split, mostly wildcards = probably over-merged) are flagged **hard**.
3. Only hard clusters are escalated to **Gemini** (`logmine/llm.py`, plain REST, no SDK). Each LLM template is validated by regex against the line it came from; output that doesn't cover its own input is rejected, not trusted.
4. The UI shows a template tree with per-template count series and robust-z spike flags.

## Results (LogHub-2k, 16 systems, 32,000 lines)

| | mean grouping accuracy |
|---|---|
| Published Drain baseline ([logparser README](https://github.com/logpai/logparser/blob/main/logparser/Drain/README.md)) | 0.8654 |
| **This Drain, per-dataset settings** | **0.8654** (matches on all 16) |
| This Drain, one generic set of masks | 0.779 |
| Hybrid with a *perfect* oracle on hard clusters | 0.879 |

- Reproducing the published column exactly is the correctness check for the parser, not an improvement.
- The per-dataset regexes/thresholds come from the logparser benchmark and were tuned on these same sets. The generic column is the realistic out-of-the-box number; Windows, Linux and Proxifier fall apart without dataset-specific masking.
- **The oracle row is an upper bound, not a Gemini result.** It replays ground-truth templates for flagged clusters. Perfect escalation would add about 1.4 points on average, concentrated in HealthApp (0.78 to 0.90), OpenStack (0.73 to 0.79), Mac (0.79 to 0.82). On the other datasets the flag heuristic misses the errors, so even a perfect LLM can't help.
- Flagged clusters are 4.5% of lines on average (1,039 LLM calls over 32,000 lines).

### Not done yet

- **No live Gemini run.** I had no API key when building this. The client is unit-tested against a fake transport only. Run `GEMINI_API_KEY=... python -m logmine.bench --llm gemini` to get the real accuracy and token counts.
- **Cost per million lines is modelled, not billed**, using placeholder prices (`--price-in/--price-out`, default $0.10/$0.40 per M tokens: set them from the current Gemini pricing page). It extrapolates the 2k-sample call rate linearly, which overstates cost because cluster counts saturate on larger logs.
- **Not run on the full ~47M-line LogHub 2.0.** Those files are multi-GB downloads from [Zenodo](https://zenodo.org/records/8196385). `scripts/run_full.py <structured.csv> <Dataset>` runs the same pipeline on them and reports throughput and GA. The 2k results do not predict 2.0 numbers; 2.0 has many more rare templates.

## Run

```bash
pip install numpy pytest
python -m pytest -q
python -m logmine.bench --llm oracle        # downloads LogHub-2k, writes docs/results.json
python -m logmine.bench --llm none --generic
python scripts/build_site.py                # -> docs/index.html
```

Credit: Drain (He et al., ICWS'17); benchmark settings and baselines from logpai/logparser (MIT); data from logpai/loghub.
