"""Fetch the LogHub 2k samples (real logs, human-verified templates)."""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/logpai/loghub/master/{d}/{d}_2k.log_structured.csv"
DATASETS = ["HDFS", "Hadoop", "Spark", "Zookeeper", "BGL", "HPC", "Thunderbird", "Windows",
            "Linux", "Android", "HealthApp", "Apache", "Proxifier", "OpenSSH", "OpenStack", "Mac"]


def fetch(name: str, cache: Path = Path("data")) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{name}_2k.csv"
    if not p.exists():
        urllib.request.urlretrieve(BASE.format(d=name), p)
    return p


def load(name: str, cache: Path = Path("data")):
    """Returns rows with Content, EventTemplate, and best-effort time key."""
    with open(fetch(name, cache), newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))
