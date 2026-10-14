#!/usr/bin/env python3
"""Fetch the latest arXiv papers, drop ones already cited in this repo, rank by LLM relevance.

Writes data/arxiv_latest_<date>.json and prints a ranked digest.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "http://export.arxiv.org/api/query"

# Window: everything since the last update (ENGINE_MASTER_SPEC.md §12 covers ~7 Oct 2026)
DATE_FROM = os.environ.get("DATE_FROM", "202610060000")
DATE_TO = os.environ.get("DATE_TO", "202610150000")

CATEGORIES = ["cs.CL", "cs.LG", "cs.DC", "cs.AI", "cs.AR", "cs.SE"]

# High-signal LLM keywords: (regex, weight)
KEYWORDS = [
    (r"\bLLM(s)?\b", 3), (r"large language model", 3), (r"language model", 2),
    (r"KV cache", 4), (r"\bKV\b", 2), (r"prefill", 3), (r"decode", 1),
    (r"speculative decoding", 4), (r"speculative", 2), (r"draft model", 3),
    (r"quantiz", 3), (r"\bGGUF\b", 3), (r"\bAWQ\b", 2), (r"\bGPTQ\b", 2),
    (r"MoE|mixture[- ]of[- ]experts|mixture of experts", 4),
    (r"expert offload", 4), (r"offload", 3), (r"prefetch", 3),
    (r"SGLang", 4), (r"vLLM", 4), (r"llama\.cpp", 4), (r"TensorRT", 3),
    (r"long[- ]context", 3), (r"attention", 2), (r"sparse attention", 3),
    (r"reasoning", 2), (r"post[- ]training", 3), (r"RLHF|RLVR|GRPO|DPO", 3),
    (r"on-policy distillation", 4), (r"distillation", 2),
    (r"RAG|retrieval", 2), (r"agent", 2), (r"inference", 2), (r"throughput", 2),
    (r"latency", 2), (r"TTFT", 3), (r"serving|scheduler", 3),
    (r"transformer", 2), (r"fine[- ]tun", 2), (r"LoRA", 2), (r"train", 1),
    (r"edge|on-device|mobile", 2), (r"\bCPU\b", 2), (r"\bGPU\b", 2),
    (r"memory bandwidth|bandwidth", 2), (r"compression", 2), (r"memory", 1),
    (r"Vulkan|Metal|WebGPU|CUDA|ROCm", 2), (r"kernel", 2),
    (r"hallucinat|safety|jailbreak", 1), (r"context window", 2),
    (r"alignment", 1), (r"multimodal|vision-language", 1), (r"token", 1),
]

TITLE_BONUS = 2  # keyword hits in title count double


def collect_existing_ids():
    ids = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules")]
        for fn in filenames:
            if not fn.endswith((".md", ".json")):
                continue
            p = os.path.join(dirpath, fn)
            if fn == "graph.json" or re.match(r"arxiv_latest_\d+\.json$", fn):
                continue
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except OSError:
                continue
            ids.update(re.findall(r"(2\d{3}\.\d{4,5})", txt))
    return ids


def parse_feed(xml):
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    out = []
    for e in entries:
        def grab(tag):
            m = re.search(r"<%s>(.*?)</%s>" % (tag, tag), e, re.S)
            return m.group(1).strip() if m else ""
        raw_id = grab("id")
        aid = re.search(r"abs/([\d.]+v?\d*)", raw_id)
        aid = aid.group(1) if aid else raw_id
        base = re.sub(r"v\d+$", "", aid)
        title = re.sub(r"\s+", " ", grab("title"))
        abstract = re.sub(r"\s+", " ", grab("summary"))
        cats = re.findall(r'term="([^"]+)"', e)
        out.append({
            "id": base,
            "title": title,
            "abstract": abstract,
            "published": grab("published")[:10],
            "updated": grab("updated")[:10],
            "categories": cats,
        })
    return out


def fetch(cat, start, max_results):
    q = "cat:%s AND submittedDate:[%s TO %s]" % (cat, DATE_FROM, DATE_TO)
    url = API + "?" + urllib.parse.urlencode({
        "search_query": q, "start": start, "max_results": max_results,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as exc:  # noqa: BLE001
            print("  retry %d after %s" % (attempt + 1, exc), file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    return ""


def score(paper):
    text = paper["title"] + " " + paper["abstract"]
    s = 0
    for pat, w in KEYWORDS:
        n_abs = len(re.findall(pat, text, re.I))
        n_title = len(re.findall(pat, paper["title"], re.I))
        s += w * (n_abs + n_title * TITLE_BONUS)
    # category affinity
    for c in paper["categories"]:
        if c in ("cs.CL", "cs.LG", "cs.DC"):
            s += 4
    return s


def main():
    existing = collect_existing_ids()
    print("existing repo arXiv IDs: %d" % len(existing), file=sys.stderr)
    papers = {}
    for cat in CATEGORIES:
        start, page, total = 0, 500, 0
        while start < 5000:
            got = parse_feed(fetch(cat, start, page))
            total += len(got)
            for p in got:
                papers[p["id"]] = p
            oldest = min((g["published"] for g in got), default="9999")
            if len(got) < page or oldest.replace("-", "") < DATE_FROM:
                break
            start += page
            time.sleep(2.0)
        print("  %s -> %d entries" % (cat, total), file=sys.stderr)
        time.sleep(1.0)
    fresh = [p for p in papers.values() if p["id"] not in existing]
    for p in fresh:
        p["score"] = score(p)
    fresh.sort(key=lambda p: (-p["score"], p["id"]))
    new = [p for p in fresh if p["score"] >= 60]
    stamp = DATE_TO[:8]
    outp = os.path.join(ROOT, "data", "arxiv_latest_%s.json" % stamp)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
    print("total unique fetched: %d | fresh: %d | kept(score>=60): %d -> %s"
          % (len(papers), len(fresh), len(new), outp), file=sys.stderr)
    for p in new[:400]:
        print("%5d  %s  %s | %s" % (p["score"], p["id"], p["published"], p["title"][:140]))


if __name__ == "__main__":
    main()
