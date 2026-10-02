"""Log parsing metrics as defined in the LogHub / logparser benchmark."""
from __future__ import annotations

from collections import defaultdict
from math import comb


def grouping_accuracy(pred: list, truth: list) -> float:
    """Share of lines whose predicted group equals their ground-truth group exactly."""
    assert len(pred) == len(truth)
    gt_groups, pr_groups = defaultdict(set), defaultdict(set)
    for i, (p, t) in enumerate(zip(pred, truth)):
        gt_groups[t].add(i)
        pr_groups[p].add(i)
    ok = 0
    for members in gt_groups.values():
        p0 = pred[next(iter(members))]
        if pr_groups[p0] == members:
            ok += len(members)
    return ok / len(pred)


def pairwise_f1(pred: list, truth: list) -> float:
    def pairs(labels):
        counts = defaultdict(int)
        for l in labels:
            counts[l] += 1
        return sum(comb(c, 2) for c in counts.values())

    joint = defaultdict(int)
    for p, t in zip(pred, truth):
        joint[(p, t)] += 1
    tp = sum(comb(c, 2) for c in joint.values())
    pp, tt = pairs(pred), pairs(truth)
    prec = tp / pp if pp else 1.0
    rec = tp / tt if tt else 1.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def parsing_accuracy(pred_templates: list[str], truth_templates: list[str]) -> float:
    """Exact template string match per line (stricter than grouping accuracy)."""
    return sum(a == b for a, b in zip(pred_templates, truth_templates)) / len(pred_templates)
