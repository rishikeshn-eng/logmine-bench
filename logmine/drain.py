"""A compact Drain (He et al., ICWS'17) implementation.

Fixed-depth parse tree: length -> first (depth-2) tokens -> leaf cluster list.
Written from the paper so we control the internals (cluster hardness signals).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

WILDCARD = "<*>"

# Generic masking applied before parsing. Deliberately dataset-agnostic: the
# published Drain numbers use per-dataset regexes, so expect small deltas.
DEFAULT_MASKS = [
    r"blk_-?\d+",
    r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?![\w.])",  # IPv4[:port]
    r"0x[0-9a-fA-F]+",
    r"(?<![\w])[+-]?\d+(?:\.\d+)?(?![\w])",
]
_MASK_RE = [re.compile(m) for m in DEFAULT_MASKS]


def compile_masks(patterns) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


def preprocess(line: str, masks=_MASK_RE) -> str:
    for m in masks:
        line = m.sub(WILDCARD, line)
    return line


def tokenize(line: str) -> list[str]:
    return line.strip().split()


@dataclass
class Cluster:
    cid: int
    template: list[str]
    line_ids: list[int] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)  # raw lines, capped

    @property
    def size(self) -> int:
        return len(self.line_ids)

    @property
    def wildcard_ratio(self) -> float:
        return sum(t == WILDCARD for t in self.template) / max(1, len(self.template))

    def text(self) -> str:
        return " ".join(self.template)


class _Node:
    __slots__ = ("children", "clusters")

    def __init__(self):
        self.children: dict[str, _Node] = {}
        self.clusters: list[Cluster] = []


def _has_digit(tok: str) -> bool:
    return any(c.isdigit() for c in tok)


class Drain:
    def __init__(self, depth: int = 4, sim_threshold: float = 0.5, max_children: int = 100,
                 keep_examples: int = 3, masks=None):
        self.depth = depth
        self.st = sim_threshold
        self.max_children = max_children
        self.keep_examples = keep_examples
        self.masks = _MASK_RE if masks is None else compile_masks(masks)
        self.root: dict[int, _Node] = {}
        self.clusters: list[Cluster] = []

    # -- tree helpers -----------------------------------------------------
    def _seq_dist(self, tpl: list[str], toks: list[str]) -> tuple[float, int]:
        same = wild = 0
        for a, b in zip(tpl, toks):
            if a == WILDCARD:
                wild += 1
            elif a == b:
                same += 1
        return same / len(tpl), wild

    def _match(self, node: _Node, toks: list[str]) -> Cluster | None:
        best, best_sim, best_wild = None, -1.0, -1
        for c in node.clusters:
            sim, wild = self._seq_dist(c.template, toks)
            if sim > best_sim or (sim == best_sim and wild > best_wild):
                best, best_sim, best_wild = c, sim, wild
        return best if best is not None and best_sim >= self.st else None

    def _search(self, toks: list[str]) -> Cluster | None:
        node = self.root.get(len(toks))
        if node is None:
            return None
        for tok in toks[: self.depth - 3]:
            nxt = node.children.get(tok) or node.children.get(WILDCARD)
            if nxt is None:
                return None
            node = nxt
        return self._match(node, toks)

    def _insert(self, c: Cluster) -> None:
        toks = c.template
        node = self.root.setdefault(len(toks), _Node())
        for tok in toks[: self.depth - 3]:
            kids = node.children
            if tok in kids:
                node = kids[tok]
            elif not _has_digit(tok):
                if WILDCARD in kids:
                    if len(kids) < self.max_children:
                        node = kids.setdefault(tok, _Node())
                    else:
                        node = kids[WILDCARD]
                elif len(kids) + 1 < self.max_children:
                    node = kids.setdefault(tok, _Node())
                else:
                    node = kids.setdefault(WILDCARD, _Node())
            else:
                node = kids.setdefault(WILDCARD, _Node())
        node.clusters.append(c)

    # -- public API ---------------------------------------------------------
    def add(self, line_id: int, raw: str) -> Cluster:
        toks = tokenize(preprocess(raw, self.masks)) or [WILDCARD]
        c = self._search(toks)
        if c is None:
            c = Cluster(len(self.clusters), list(toks))
            self.clusters.append(c)
            self._insert(c)
        else:
            c.template = [a if a == b else WILDCARD for a, b in zip(c.template, toks)]
        c.line_ids.append(line_id)
        if len(c.examples) < self.keep_examples:
            c.examples.append(raw)
        return c

    def fit(self, lines) -> list[int]:
        """Parse lines; returns the cluster id for each line."""
        return [self.add(i, l).cid for i, l in enumerate(lines)]
