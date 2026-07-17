"""Queue runner.

  python src/run.py                 process every pending case (priority first)
  python src/run.py --now PATH|ID   run one case immediately (external files are
                                    copied into cases/ first)
  python src/run.py --force         also re-run cases whose config/prompts changed
  python src/run.py --list          show queue status and exit
"""
import shutil
import sys
import time
import traceback
from pathlib import Path

from common import CASES_DIR, RESULTS_DIR, RUNS_DIR, config_hash, read_json, write_json
from pipeline import STAGES, new_state


def load_cases() -> list[dict]:
    cases = [read_json(p) for p in sorted(CASES_DIR.glob("*.json"))]
    return sorted(cases, key=lambda c: (not c.get("priority", False), c["id"]))


def result_path(case_id: str) -> Path:
    return RESULTS_DIR / f"{case_id}.json"


def status(case: dict, chash: str) -> str:
    path = result_path(case["id"])
    if not path.exists():
        return "pending"
    result = read_json(path)
    if len(result.get("stages_done", [])) < len(STAGES):
        return "partial"
    if result.get("config_hash") != chash:
        return "stale"
    return "done"


def run_case(case: dict, chash: str) -> None:
    path = result_path(case["id"])
    state = read_json(path) if path.exists() else new_state(case)
    if state.get("config_hash") != chash:  # config/prompts changed -> start over
        state = new_state(case)
    for name, fn in STAGES:
        if name in state["stages_done"]:
            continue
        t0 = time.time()
        fn(case, state)
        state["stages_done"].append(name)
        state.setdefault("stage_seconds", {})[name] = round(time.time() - t0, 1)
        write_json(path, state)  # checkpoint after every stage
        print(f"    {name:16s} {state['stage_seconds'][name]:6.1f}s")


def main() -> None:
    args = sys.argv[1:]
    chash = config_hash()

    if "--now" in args:
        target = args[args.index("--now") + 1]
        path = Path(target)
        if path.exists():  # external file: adopt it into the queue
            dest = CASES_DIR / path.name
            if path.resolve() != dest.resolve():
                shutil.copy(path, dest)
            case = read_json(dest)
        else:
            case = next(c for c in load_cases() if c["id"] == target)
        queue = [case]
    else:
        cases = load_cases()
        wanted = (("pending", "partial", "stale", "done") if "--force" in args
                  else ("pending", "partial"))
        queue = [c for c in cases if status(c, chash) in wanted]

    if "--list" in args:
        for c in load_cases():
            print(f"{status(c, chash):8s} {c['id']}  [{c.get('source', '?')}]"
                  f"{'  PRIORITY' if c.get('priority') else ''}")
        return

    if not queue:
        print("Queue empty — nothing pending.")
        return

    # --force / --now mean "run it again": reset state so every stage re-executes
    # (the response cache makes unchanged LLM calls free anyway)
    if "--force" in args or "--now" in args:
        for case in queue:
            write_json(result_path(case["id"]), new_state(case))

    print(f"Processing {len(queue)} case(s), config {chash}")
    manifest = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_hash": chash,
                "cases": [c["id"] for c in queue]}
    started = time.time()
    failures = []
    # Stage-major order: one model stays resident across all cases per stage.
    for name, fn in STAGES:
        for case in queue:
            path = result_path(case["id"])
            state = read_json(path) if path.exists() else new_state(case)
            if state.get("config_hash") != chash:
                state = new_state(case)
            if name in state["stages_done"]:
                continue
            print(f"  {case['id']}: {name}")
            try:
                t0 = time.time()
                fn(case, state)
                state["stages_done"].append(name)
                state.setdefault("stage_seconds", {})[name] = round(time.time() - t0, 1)
                write_json(path, state)
            except Exception:
                failures.append((case["id"], name))
                print(f"  !! {case['id']} failed at {name}\n{traceback.format_exc()}")

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["seconds"] = round(time.time() - started, 1)
    manifest["failures"] = failures
    write_json(RUNS_DIR / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json", manifest)
    print(f"Done in {manifest['seconds']}s; failures: {failures or 'none'}")


if __name__ == "__main__":
    main()
