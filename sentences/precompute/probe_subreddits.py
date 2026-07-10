"""Scan reddit-title-body for our target subreddits and sample real rows."""

from collections import Counter
from datasets import load_dataset

TARGETS = {
    "fitness", "nutrition", "loseit", "gainit", "bodybuilding", "supplements",
    "personalfinance", "investing", "financialindependence", "frugal",
    "health", "keto", "intermittentfasting", "running", "weightlifting",
    "askscience", "explainlikeimfive", "getmotivated", "productivity",
}

ds = load_dataset("sentence-transformers/reddit-title-body", split="train", streaming=True)

counts = Counter()
samples = {}
scanned = 0
for row in ds:
    scanned += 1
    sub = row.get("subreddit", "")
    if sub in TARGETS:
        counts[sub] += 1
        if sub not in samples:
            body = str(row.get("body", "")).replace("\n", " ")
            samples[sub] = body[:120]
    if scanned >= 400000:
        break

print(f"scanned {scanned} rows\n")
print("target subreddit hits (in first 400k):")
for sub, n in counts.most_common():
    print(f"  {n:>6}  r/{sub}")
print(f"\ntotal target hits: {sum(counts.values())}")
print("\nsamples:")
for sub, text in list(samples.items())[:8]:
    print(f"  r/{sub}: {text}")
