import json

import numpy as np

from logmine import metrics
from logmine.drain import Drain
from logmine.hybrid import run_hybrid
from logmine.llm import GeminiLLM, OracleLLM, validate
from logmine.spikes import spikes

LINES = (["Connection from 10.0.0.%d port %d" % (i, 1000 + i) for i in range(20)]
         + ["Disk /dev/sda%d is full" % i for i in range(15)]
         + ["Started service alpha"])


def test_drain_groups_by_event():
    d = Drain()
    cids = d.fit(LINES)
    assert len(set(cids[:20])) == 1 and len(set(cids[20:35])) == 1
    assert cids[0] != cids[20] != cids[-1]
    assert d.clusters[cids[0]].text() == "Connection from <*> port <*>"


def test_metrics_perfect_and_split():
    truth = [0, 0, 1, 1, 1]
    assert metrics.grouping_accuracy(truth, truth) == 1.0
    assert metrics.grouping_accuracy([0, 1, 2, 2, 2], truth) == 0.6  # group 0 split, group 1 intact
    assert metrics.pairwise_f1(truth, truth) == 1.0


def test_validate_rejects_template_that_misses_lines():
    assert validate("Disk <*> is full", ["Disk /dev/sda1 is full"])
    assert not validate("Disk <*> is empty", ["Disk /dev/sda1 is full"])
    assert not validate("<*>", ["anything"])


def _fake(template):
    def transport(body):
        return {"candidates": [{"content": {"parts": [{"text": json.dumps({"template": template})}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10}}
    return transport


def test_gemini_client_parses_and_counts_tokens():
    g = GeminiLLM(transport=_fake("Disk <*> is full"))
    assert g.template_for(["Disk /dev/sda1 is full"]) == "Disk <*> is full"
    assert (g.usage.calls, g.usage.tokens_in, g.usage.tokens_out) == (1, 100, 10)


def test_gemini_bad_output_is_rejected_not_trusted():
    g = GeminiLLM(transport=_fake("Totally unrelated"))
    assert g.template_for(["Disk /dev/sda1 is full"]) is None
    assert g.usage.rejected == 1


def test_hybrid_oracle_fixes_overmerged_cluster():
    lines = ["Task a failed", "Task b failed", "Task c succeeded"] * 3
    truth = ["Task <*> failed", "Task <*> failed", "Task <*> succeeded"] * 3
    g0, _, s0, _ = run_hybrid(lines, st=0.3)
    g1, t1, s1, _ = run_hybrid(lines, OracleLLM(), truth, st=0.3)
    tg = [0 if "failed" in t else 1 for t in truth]
    assert metrics.grouping_accuracy(g1, tg) == 1.0
    assert t1 == truth


def test_spikes_flags_outlier_window_only():
    s = np.array([5, 6, 5, 4, 6, 5, 90, 5, 6, 5])
    assert spikes(s) == [6]
    assert spikes(np.array([5] * 10)) == []
