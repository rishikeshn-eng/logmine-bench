"""Drain first; escalate only clusters flagged hard; merge by final template."""
from __future__ import annotations

from .drain import Drain, Cluster
from .llm import template_regex


def is_hard(c: Cluster, min_support: int = 2, max_wildcards: float = 0.6) -> bool:
    """Cheap signals that Drain probably got this cluster wrong.

    - tiny support: likely an over-split fragment of a bigger event
    - mostly wildcards: likely an over-merge of unrelated events
    """
    return c.size < min_support or (c.wildcard_ratio > max_wildcards and len(c.template) > 2)


def _escalate(c: Cluster, lines, llm, truth_templates, max_calls: int) -> dict[int, str]:
    """Cover a hard cluster's lines with LLM templates, one call per uncovered line.

    Each returned template is applied to every remaining line it matches, so a
    cluster Drain over-merged gets split and one it over-split gets unified.
    """
    remaining = list(c.line_ids)
    assigned: dict[int, str] = {}
    calls = 0
    while remaining and calls < max_calls:
        lid = remaining[0]
        hint = truth_templates[lid] if truth_templates else None
        tpl = llm.template_for([lines[lid]], hint)
        calls += 1
        if not tpl:
            remaining.pop(0)
            continue
        rx = template_regex(tpl)
        hit = [i for i in remaining if rx.match(lines[i])]
        for i in hit:
            assigned[i] = tpl
        remaining = [i for i in remaining if i not in assigned]
    return assigned


def run_hybrid(lines: list[str], llm=None, truth_templates: list[str] | None = None,
               min_support: int = 2, max_wildcards: float = 0.6, st: float = 0.5, depth: int = 4,
               masks=None, max_calls_per_cluster: int = 25):
    """Returns (group_ids_per_line, templates_per_line, stats, drain).

    truth_templates is only consulted by OracleLLM (as the hint it replays).
    """
    d = Drain(depth=depth, sim_threshold=st, masks=masks)
    cids = d.fit(lines)
    by_cluster = {c.cid: c.text() for c in d.clusters}
    per_line = [by_cluster[cid] for cid in cids]
    hard = [c for c in d.clusters if is_hard(c, min_support, max_wildcards)]
    if llm is not None:
        for c in hard:
            for lid, tpl in _escalate(c, lines, llm, truth_templates, max_calls_per_cluster).items():
                per_line[lid] = tpl
    ids: dict[str, int] = {}
    groups = [ids.setdefault(t, len(ids)) for t in per_line]
    stats = {"lines": len(lines), "drain_clusters": len(d.clusters), "hard_clusters": len(hard),
             "hard_lines": sum(c.size for c in hard)}
    return groups, per_line, stats, d
