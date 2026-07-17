"""Shared plumbing: paths, config, content-addressed cache, call log."""
import hashlib
import json
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = EVAL_DIR.parent
CASES_DIR = EVAL_DIR / "cases"
RESULTS_DIR = EVAL_DIR / "results"
RUNS_DIR = EVAL_DIR / "runs"
CACHE_DIR = EVAL_DIR / "cache"
PROMPTS_DIR = EVAL_DIR / "prompts"

for d in (CASES_DIR, RESULTS_DIR, RUNS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

CONFIG = json.loads((EVAL_DIR / "config.json").read_text(encoding="utf-8"))


def sha256(obj) -> str:
    payload = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_prompt(name: str) -> dict:
    """Role prompt file -> {'system': ..., 'user': ...} (split on SYSTEM:/USER: markers)."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    system, _, user = text.partition("USER:")
    system = system.replace("SYSTEM:", "", 1).strip()
    return {"system": system, "user": user.strip()}


def fill(template: str, **tokens) -> str:
    out = template
    for key, value in tokens.items():
        out = out.replace(f"<<{key.upper()}>>", str(value))
    return out


def config_hash() -> str:
    """Hash of config + all role prompts: results carry it so stale runs are detectable."""
    prompts = {p.name: p.read_text(encoding="utf-8") for p in sorted(PROMPTS_DIR.glob("*.md"))}
    return sha256({"config": CONFIG, "prompts": prompts})[:16]


def cache_get(key: str):
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def cache_put(key: str, value) -> None:
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def log_call(record: dict) -> None:
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with (RUNS_DIR / "calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
