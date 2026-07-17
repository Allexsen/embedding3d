"""Frontier audit: re-judge a seeded random sample of completed cases on a
big external model and report agreement with the local judge.

  python src/audit.py --endpoint http://localhost:8080/v1 --model glm-5.2 \
                      [--track main|convo] [--frac 0.1] [--seed 7]

Works against any OpenAI-compatible /chat/completions endpoint: Colibri
(`coli serve`), Ollama (incl. its -cloud models), or a hosted API (set
AUDIT_API_KEY for Bearer auth). The auditor re-scores THE SAME artifacts,
blind — it never sees the local verdicts. Output: results/audit/<name>.md
with per-stage agreement; disagreements are listed, never "corrected".
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

from common import EVAL_DIR, RESULTS_DIR, load_prompt, fill, read_json, write_json
from pipeline import STAGES as MAIN_STAGES, VERDICT_SCORE

AUDIT_DIR = RESULTS_DIR / "audit"
AUDIT_DIR.mkdir(exist_ok=True)


def call(endpoint: str, model: str, system: str, user: str, timeout: int = 3600) -> str:
    import os
    headers = {"Content-Type": "application/json"}
    if os.environ.get("AUDIT_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['AUDIT_API_KEY']}"
    resp = requests.post(f"{endpoint.rstrip('/')}/chat/completions", headers=headers,
                         json={"model": model, "temperature": 0,
                               "messages": [{"role": "system", "content": system},
                                            {"role": "user", "content": user}]},
                         timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_coverage(text: str, n_facets: int) -> list | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        entries = json.loads(match.group(0))["coverage"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    out = []
    for i in range(n_facets):
        e = next((x for x in entries if x.get("facet") == i + 1), None)
        out.append(VERDICT_SCORE.get((e or {}).get("verdict", "no"), 0.0))
    return out


def main() -> None:
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    endpoint = args["--endpoint"]
    model = args["--model"]
    track = args.get("--track", "main")
    frac = float(args.get("--frac", 0.1))
    rng = np.random.default_rng(int(args.get("--seed", 7)))

    results_dir = RESULTS_DIR if track == "main" else EVAL_DIR / "results_convo"
    dev_ids = {read_json(p)["id"]
               for d in ("cases", "convos") if (EVAL_DIR / d).exists()
               for p in (EVAL_DIR / d).glob("*.json") if read_json(p).get("dev")}
    done = [read_json(p) for p in sorted(results_dir.glob("*.json"))
            if p.suffix == ".json" and p.stem not in ("summary",)
            and not p.name.startswith(("REPORT", "audit"))]
    done = [r for r in done if isinstance(r, dict) and r.get("case_id")
            and r["case_id"] not in dev_ids and not r.get("rejected")
            and r.get("facets")]
    if not done:
        print("Nothing auditable.")
        return

    k = max(1, round(len(done) * frac))
    sample = list(rng.choice(len(done), size=k, replace=False))
    prompt = load_prompt("judge_facets")

    rows, exact, total, maes = [], 0, 0, []
    for idx in sample:
        r = done[int(idx)]
        facet_block = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(r["facets"]))
        for arm, slot in r["arms"].items():
            if "answer_coverage" not in slot:
                continue
            t0 = time.time()
            reply = call(endpoint, model, prompt["system"],
                         fill(prompt["user"], facets=facet_block, text=slot["answer"]))
            got = parse_coverage(reply, len(r["facets"]))
            if got is None:
                rows.append({"case": r["case_id"], "arm": arm, "error": "unparseable"})
                continue
            local = slot["answer_coverage"]["per_facet"]
            agree = sum(1 for a, b in zip(local, got) if a == b)
            exact += agree
            total += len(local)
            maes.append(abs(np.mean(local) - np.mean(got)))
            rows.append({"case": r["case_id"], "arm": arm, "local": local,
                         "auditor": got, "facet_agreement": f"{agree}/{len(local)}",
                         "seconds": round(time.time() - t0, 1)})
            print(f"  {r['case_id']}/{arm}: {agree}/{len(local)} agree "
                  f"({rows[-1]['seconds']}s)")

    name = f"audit-{track}-{model.replace('/', '_').replace(':', '_')}"
    summary = {
        "endpoint": endpoint, "model": model, "track": track,
        "sampled_cases": k, "of": len(done),
        "facet_verdict_agreement": round(exact / max(1, total), 4),
        "coverage_score_mae": round(float(np.mean(maes)), 4) if maes else None,
        "rows": rows,
    }
    write_json(AUDIT_DIR / f"{name}.json", summary)
    lines = [f"# Frontier audit — {model} on {track} track",
             f"- sampled {k}/{len(done)} cases (seeded)",
             f"- facet-verdict agreement: {summary['facet_verdict_agreement']:.1%}",
             f"- coverage-score MAE: {summary['coverage_score_mae']}",
             "", "Disagreements are data, not errors to fix — see the JSON for rows."]
    (AUDIT_DIR / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
