"""LLM escalation for hard clusters: Gemini over plain REST, plus an offline oracle.

`OracleLLM` reads ground-truth templates. It is NOT a Gemini result; it exists
to bound what perfect escalation could buy so the cost/accuracy trade-off can be
studied without an API key.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass

from .drain import WILDCARD

PROMPT = """You are a log template extractor. Replace every variable part (numbers, ids, ips, paths, \
hostnames, user names, timestamps) in the log lines with {wc} and keep the constant words verbatim. \
The lines below are instances of ONE event type. Reply with JSON only: {{"template": "..."}}.

Lines:
{lines}
"""



def _post_json(url: str, key: str, body: dict, tries: int = 5) -> dict:
    """POST with exponential backoff on 429/5xx/network errors (Gemini rate-limits free keys)."""
    import time
    import urllib.error
    data = json.dumps(body).encode()
    for i in range(tries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
        time.sleep(min(60, 2 ** (i + 1)))


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def template_regex(template: str) -> re.Pattern:
    parts = [re.escape(p) for p in template.split(WILDCARD)]
    return re.compile(r"^\s*" + r".*?".join(parts) + r"\s*$")


def validate(template: str, examples: list[str]) -> bool:
    """Reject LLM output that does not actually cover the lines it was derived from."""
    if not template.strip() or template.strip() == WILDCARD:
        return False
    rx = template_regex(template)
    return all(rx.match(e) for e in examples)


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    rejected: int = 0


class GeminiLLM:
    """Minimal Gemini client (generateContent REST). Needs GEMINI_API_KEY."""

    def __init__(self, model: str | None = None, api_key: str | None = None, transport=None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.key and transport is None:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.transport = transport or self._http
        self.usage = Usage()

    def _http(self, body: dict) -> dict:
        return _post_json(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", self.key, body)

    def template_for(self, examples: list[str], truth_hint: str | None = None) -> str | None:
        prompt = PROMPT.format(wc=WILDCARD, lines="\n".join(examples))
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
        resp = self.transport(body)
        self.usage.calls += 1
        meta = resp.get("usageMetadata") or {}
        self.usage.tokens_in += meta.get("promptTokenCount", approx_tokens(prompt))
        try:
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
            tpl = json.loads(text)["template"]
        except (KeyError, IndexError, ValueError, TypeError):
            self.usage.rejected += 1
            return None
        self.usage.tokens_out += meta.get("candidatesTokenCount", approx_tokens(text))
        if not validate(tpl, examples):
            self.usage.rejected += 1
            return None
        return tpl


class OracleLLM:
    """Perfect escalation (ground truth). Upper bound only. Costs are modelled, not billed."""

    def __init__(self):
        self.usage = Usage()

    def template_for(self, examples: list[str], truth_hint: str | None = None) -> str | None:
        prompt = PROMPT.format(wc=WILDCARD, lines="\n".join(examples))
        self.usage.calls += 1
        self.usage.tokens_in += approx_tokens(prompt)
        self.usage.tokens_out += approx_tokens(truth_hint or "")
        return truth_hint


def cost_usd(u: Usage, price_in_per_m: float, price_out_per_m: float) -> float:
    return u.tokens_in / 1e6 * price_in_per_m + u.tokens_out / 1e6 * price_out_per_m
