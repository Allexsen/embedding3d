"""Probe candidate HF datasets for the four corpus sources.

Streams a few rows from each candidate, printing keys + a snippet, so we can
pin working dataset IDs and field names before writing the real pipeline.
"""

from datasets import load_dataset

CANDIDATES = [
    # (label, dataset id, config, split)
    ("eli5", "sentence-transformers/eli5", None, "train"),
    ("social", "webis/tldr-17", None, "train"),
    ("arxiv", "gfissore/arxiv-abstracts-2021", None, "train"),
    ("wikipedia", "wikimedia/wikipedia", "20231101.en", "train"),
]


def probe(label, dataset_id, config, split):
    print(f"\n=== {label}: {dataset_id} ===")
    try:
        ds = load_dataset(dataset_id, config, split=split, streaming=True)
        for i, row in enumerate(ds):
            if i >= 2:
                break
            keys = list(row.keys())
            print(f"  keys: {keys}")
            for key in keys[:6]:
                value = str(row[key]).replace("\n", " ")[:110]
                print(f"    {key}: {value}")
        print(f"  OK: {label}")
    except Exception as error:
        print(f"  FAILED: {type(error).__name__}: {str(error)[:220]}")


for candidate in CANDIDATES:
    probe(*candidate)
