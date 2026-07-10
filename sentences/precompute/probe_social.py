"""Probe alternative social-media datasets (webis/tldr-17 is script-based, dead)."""

from datasets import load_dataset

CANDIDATES = [
    ("geclm-fitness", "HuggingFaceGECLM/REDDIT_comments", "fitness", "train"),
    ("geclm-personalfinance", "HuggingFaceGECLM/REDDIT_comments", "personalfinance", "train"),
    ("reddit-title-body", "sentence-transformers/reddit-title-body", None, "train"),
    ("tldr17-parquet", "parquet", None, "train"),
]


def probe(label, dataset_id, config, split):
    print(f"\n=== {label} ===")
    try:
        if label == "tldr17-parquet":
            ds = load_dataset(
                "parquet",
                data_files="hf://datasets/webis/tldr-17@refs/convert/parquet/default/train/0000.parquet",
                split=split,
                streaming=True,
            )
        else:
            ds = load_dataset(dataset_id, config, split=split, streaming=True)
        for i, row in enumerate(ds):
            if i >= 2:
                break
            keys = list(row.keys())
            print(f"  keys: {keys}")
            for key in keys[:6]:
                value = str(row[key]).replace("\n", " ")[:100]
                print(f"    {key}: {value}")
        print(f"  OK: {label}")
    except Exception as error:
        print(f"  FAILED: {type(error).__name__}: {str(error)[:200]}")


for candidate in CANDIDATES:
    probe(*candidate)
