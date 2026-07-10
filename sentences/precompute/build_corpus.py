"""Build the register-diverse corpus for Embedding3D · Sentences.

Four sources, streamed from HuggingFace, cleaned to 15-60 word passages, then
interleaved round-robin so every dataset tier is a prefix with the full
register spread. Emits corpus.jsonl (text, source) + a stats report.

  academic     arxiv abstracts (physics/cs/q-bio/econ/stat)
  encyclopedic wikipedia lead paragraphs
  casual-expl  ELI5 answers (earnest plain-words explanation)
  casual-anec  reddit posts from health/fitness/finance/lifestyle subreddits

Run:
  python build_corpus.py --per-source 25000 --out corpus.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).parent

SOURCES = ["arxiv", "wikipedia", "eli5", "social"]  # round-robin order

MIN_WORDS = 15
MAX_WORDS = 60

# arXiv category prefixes we keep — spread across sciences so registers
# compete over overlapping subjects (physics/bio/cs/econ), not just math.
ARXIV_CATS = ("physics", "cond-mat", "astro-ph", "q-bio", "cs.", "stat.",
              "econ.", "eess.", "math.NA", "nlin", "quant-ph")

SOCIAL_SUBREDDITS = {
    # fitness / health / nutrition
    "fitness", "loseit", "gainit", "bodybuilding", "supplements", "nutrition",
    "keto", "intermittentfasting", "running", "weightlifting", "xxfitness",
    "advancedfitness", "flexibility", "bodyweightfitness", "health", "sleep",
    # finance
    "personalfinance", "investing", "financialindependence", "frugal",
    "stocks", "budget", "povertyfinance",
    # lifestyle / self-improvement / casual science
    "askscience", "explainlikeimfive", "getmotivated", "productivity",
    "selfimprovement", "decidingtobebetter", "getdisciplined", "cooking",
    "nootropics", "biohackers", "science", "askscience",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_RE = re.compile(r"[*_`>#~]|\[|\]|\(http[^)]*\)")
WS_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
LATEX_RE = re.compile(r"\$[^$]*\$")
# Wikipedia leaves empty pronunciation/IPA stubs after link stripping: "(; )", "( )"
PAREN_STUB_RE = re.compile(r"\(\s*[;,]?\s*\)")
SPACE_PUNCT_RE = re.compile(r"\s+([,.;:)])")


def clean(text, drop_markdown=True):
    text = URL_RE.sub(" ", text)
    text = LATEX_RE.sub(" ", text)
    if drop_markdown:
        text = MD_RE.sub(" ", text)
    text = text.replace("&gt;", " ").replace("&lt;", " ").replace("&amp;", "&")
    text = PAREN_STUB_RE.sub(" ", text)
    text = SPACE_PUNCT_RE.sub(r"\1", text)
    return WS_RE.sub(" ", text).strip()


def to_passage(text, drop_markdown=True):
    """First 1-3 sentences that land inside the 15-60 word window, else None."""
    text = clean(text, drop_markdown=drop_markdown)
    if not text:
        return None
    sentences = SENT_SPLIT_RE.split(text)
    passage = ""
    for sentence in sentences[:3]:
        candidate = (passage + " " + sentence).strip() if passage else sentence
        words = candidate.split()
        if len(words) > MAX_WORDS:
            if not passage and len(sentence.split()) >= MIN_WORDS:
                return " ".join(sentence.split()[:MAX_WORDS])
            break
        passage = candidate
        if len(words) >= MIN_WORDS:
            break
    words = passage.split()
    if MIN_WORDS <= len(words) <= MAX_WORDS and passage[-1:] in ".!?\"'":
        return passage
    if MIN_WORDS <= len(words) <= MAX_WORDS:
        return passage
    return None


def dedup_key(passage):
    return " ".join(passage.lower().split()[:12])


def collect_arxiv(target):
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    seen, out = set(), []
    for row in ds:
        raw_cats = row.get("categories") or ""
        cats = raw_cats if isinstance(raw_cats, list) else raw_cats.split()
        if not any(cat.startswith(c) for cat in cats for c in ARXIV_CATS):
            continue
        passage = to_passage(row.get("abstract", ""), drop_markdown=False)
        if not passage:
            continue
        key = dedup_key(passage)
        if key in seen:
            continue
        seen.add(key)
        out.append(passage)
        if len(out) >= target:
            break
    return out


def collect_wikipedia(target):
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    seen, out = set(), []
    for row in ds:
        passage = to_passage(row.get("text", ""), drop_markdown=False)
        if not passage:
            continue
        key = dedup_key(passage)
        if key in seen:
            continue
        seen.add(key)
        out.append(passage)
        if len(out) >= target:
            break
    return out


def collect_eli5(target):
    ds = load_dataset("sentence-transformers/eli5", split="train", streaming=True)
    seen, out = set(), []
    for row in ds:
        passage = to_passage(row.get("answer", ""))
        if not passage:
            continue
        key = dedup_key(passage)
        if key in seen:
            continue
        seen.add(key)
        out.append(passage)
        if len(out) >= target:
            break
    return out


def collect_social(target, row_budget):
    ds = load_dataset("sentence-transformers/reddit-title-body", split="train", streaming=True)
    seen, out = set(), []
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > row_budget:
            break
        if (row.get("subreddit") or "").lower() not in SOCIAL_SUBREDDITS:
            continue
        # keep slang/typos: only strip urls + markdown, no aggressive cleanup
        passage = to_passage(row.get("body", ""))
        if not passage:
            continue
        key = dedup_key(passage)
        if key in seen:
            continue
        seen.add(key)
        out.append(passage)
        if len(out) >= target:
            break
    return out, scanned


COLLECTORS = {
    "arxiv": ("academic", collect_arxiv),
    "wikipedia": ("encyclopedic", collect_wikipedia),
    "eli5": ("casual-explanatory", collect_eli5),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-source", type=int, default=25000)
    parser.add_argument("--social-row-budget", type=int, default=6_000_000)
    parser.add_argument("--out", default="corpus.jsonl")
    args = parser.parse_args()

    collected = {}
    for name in ("arxiv", "wikipedia", "eli5"):
        register, fn = COLLECTORS[name]
        print(f"collecting {name} ({register}) …", flush=True)
        collected[name] = fn(args.per_source)
        print(f"  {name}: {len(collected[name])} passages", flush=True)

    print("collecting social (casual-anecdotal) …", flush=True)
    social, scanned = collect_social(args.per_source, args.social_row_budget)
    collected["social"] = social
    print(f"  social: {len(social)} passages (scanned {scanned} rows)", flush=True)

    # round-robin interleave so every prefix keeps the register spread
    limit = min(len(collected[s]) for s in SOURCES)
    print(f"\nbalancing to {limit} per source ({limit * len(SOURCES)} total)")

    registers = {
        "arxiv": "academic", "wikipedia": "encyclopedic",
        "eli5": "casual-explanatory", "social": "casual-anecdotal",
    }
    out_path = HERE / args.out
    written = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for i in range(limit):
            for name in SOURCES:
                passage = collected[name][i]
                handle.write(json.dumps({
                    "text": passage,
                    "source": name,
                    "register": registers[name],
                }, ensure_ascii=False) + "\n")
                written += 1

    print(f"wrote {out_path}: {written} passages")
    write_report(collected, limit, registers)


def write_report(collected, limit, registers):
    lines = ["# Corpus stats\n"]
    for name in SOURCES:
        passages = collected[name]
        word_counts = [len(p.split()) for p in passages[:limit]]
        avg = sum(word_counts) / max(1, len(word_counts))
        lines.append(f"## {name} ({registers[name]}) — {limit} used of {len(passages)} collected, avg {avg:.1f} words\n")
        for sample in passages[:12]:
            lines.append(f"- {sample}")
        lines.append("")
    (HERE / "corpus_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote corpus_report.md (review the samples per source)")


if __name__ == "__main__":
    main()
