"""Aggregate finished results into a human-readable report + machine-readable summary.

  python src/report.py

Primary metric: facet coverage of the ANSWER (0..1), per arm, paired across cases.
Deltas B-A and C-A with a paired bootstrap 95% CI and a paired permutation p-value.
Everything stratified by severity (how little of the true intent the prompt conveyed).
"""
import json
from pathlib import Path

import numpy as np

from common import RESULTS_DIR, read_json
from pipeline import STAGES

RNG = np.random.default_rng(12345)


def paired_bootstrap(delta: np.ndarray, n: int = 10000) -> tuple:
    if len(delta) == 0:
        return (float("nan"), float("nan"))
    means = [RNG.choice(delta, len(delta), replace=True).mean() for _ in range(n)]
    return (round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4))


def perm_p(delta: np.ndarray, n: int = 10000) -> float:
    """Two-sided paired permutation: randomly flip the sign of each pair's delta."""
    if len(delta) == 0:
        return float("nan")
    observed = abs(delta.mean())
    hits = 0
    for _ in range(n):
        signs = RNG.choice([-1, 1], len(delta))
        if abs((delta * signs).mean()) >= observed:
            hits += 1
    return round((hits + 1) / (n + 1), 4)


def summarize(values: np.ndarray) -> dict:
    return {"mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "n": int(len(values))}


def main() -> None:
    from common import CASES_DIR
    dev_ids = {read_json(p)["id"] for p in CASES_DIR.glob("*.json")
               if read_json(p).get("dev")}
    done, dev_skipped = [], []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name in ("REPORT.md", "summary.json"):
            continue
        r = read_json(path)
        if len(r.get("stages_done", [])) < len(STAGES):
            continue
        # dev cases tuned the prompts — they never enter headline statistics
        (dev_skipped if r["case_id"] in dev_ids else done).append(r)

    if not done:
        if dev_skipped:
            print(f"Only dev-set results exist ({len(dev_skipped)}); no headline cases yet.")
        else:
            print("No completed results yet.")
        return

    ids = [r["case_id"] for r in done]
    ans = {arm: np.array([r["arms"][arm]["answer_coverage"]["score"] for r in done])
           for arm in ("A", "B", "C")}
    rec = {arm: np.array([r["arms"][arm]["recovered_coverage"]["score"] for r in done])
           for arm in ("A", "B", "C")}
    severity = np.array([r["severity"] for r in done])
    overhead = {arm: np.array([r["arms"][arm].get("overhead_tokens", 0) for r in done])
                for arm in ("A", "B", "C")}
    paid = {arm: np.array([r["arms"][arm]["answer_tokens"]["in"]
                           + r["arms"][arm]["answer_tokens"]["out"] for r in done])
            for arm in ("A", "B", "C")}
    b_fidelity = np.array([1 if done[i]["arms"]["B"]["fidelity"]["same"] else 0
                           for i in range(len(done))])

    lines = [f"# Eval report — {len(done)} completed case(s)", "",
             f"cases: {', '.join(ids)}", "",
             "## Primary: answer facet-coverage (0..1), paired", ""]
    for arm in ("A", "B", "C"):
        s = summarize(ans[arm])
        lines.append(f"- arm {arm}: mean {s['mean']}  median {s['median']}")
    lines.append("")

    summary = {"n": len(done), "cases": ids, "answer_coverage": {}, "deltas": {}}
    for arm in ("A", "B", "C"):
        summary["answer_coverage"][arm] = summarize(ans[arm])

    lines.append("## Deltas vs arm A (positive = better than the user's raw prompt)\n")
    for arm in ("B", "C"):
        d = ans[arm] - ans["A"]
        ci, p = paired_bootstrap(d), perm_p(d)
        lines += [f"### {arm} − A  (answer coverage)",
                  f"- mean delta: {round(float(d.mean()), 4)}",
                  f"- 95% CI: [{ci[0]}, {ci[1]}]",
                  f"- permutation p: {p}", ""]
        summary["deltas"][f"{arm}-A"] = {"mean": round(float(d.mean()), 4),
                                         "ci95": ci, "perm_p": p}

    # severity stratification: does the effect grow as the prompt conveys less?
    lines.append("## By severity tercile (severity = 1 − how much the raw prompt conveyed)\n")
    order = np.argsort(severity)
    thirds = np.array_split(order, 3) if len(done) >= 3 else [order]
    band_names = ["low", "mid", "high"][:len(thirds)]
    summary["severity_bands"] = {}
    for name, band in zip(band_names, thirds):
        if len(band) == 0:
            continue
        row = {"n": int(len(band)),
               "severity_mean": round(float(severity[band].mean()), 3)}
        seg = [f"### {name} severity  (n={len(band)}, mean severity {row['severity_mean']})"]
        for arm in ("A", "B", "C"):
            m = round(float(ans[arm][band].mean()), 4)
            row[f"cov_{arm}"] = m
            seg.append(f"- arm {arm} answer coverage: {m}")
        seg.append(f"- C − A: {round(float((ans['C'] - ans['A'])[band].mean()), 4)}")
        seg.append(f"- B − A: {round(float((ans['B'] - ans['A'])[band].mean()), 4)}")
        lines += seg + [""]
        summary["severity_bands"][name] = row

    c_noop = sum(1 for r in done if r["arms"]["C"].get("noop"))
    lines.append("## Secondary channels\n")
    lines.append(f"- C no-op gate (mirror left the prompt untouched): {c_noop}/{len(done)}")
    lines.append(f"- B fidelity (rewrite preserved meaning): {int(b_fidelity.sum())}/{len(done)}")
    for arm in ("A", "B", "C"):
        rc = summarize(rec[arm])
        lines.append(f"- recovered-intent coverage, arm {arm}: mean {rc['mean']}")
    for arm in ("A", "B", "C"):
        lines.append(f"- tokens: arm {arm} overhead {int(overhead[arm].mean())} + "
                     f"answer {int(paid[arm].mean())} (mean)")
    lines.append("")
    summary["b_fidelity_preserved"] = int(b_fidelity.sum())
    summary["recovered_coverage"] = {a: summarize(rec[a]) for a in ("A", "B", "C")}
    summary["tokens_mean"] = {a: {"overhead": int(overhead[a].mean()),
                                  "answer": int(paid[a].mean())} for a in ("A", "B", "C")}

    # flat per-case table for ad-hoc analysis (pandas-ready)
    import csv
    with (RESULTS_DIR / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "source", "severity", "cov_A", "cov_B", "cov_C",
                    "delta_B", "delta_C", "b_fidelity", "c_noop",
                    "rec_A", "rec_B", "rec_C", "ans_tokens_A", "ans_tokens_B",
                    "ans_tokens_C", "cos_rec_A", "cos_rec_B", "cos_rec_C",
                    "overlap_B_vs_A", "overlap_C_vs_A"])
        for r in done:
            a = r["arms"]
            w.writerow([
                r["case_id"], r["case_id"].split("-")[2], r["severity"],
                a["A"]["answer_coverage"]["score"], a["B"]["answer_coverage"]["score"],
                a["C"]["answer_coverage"]["score"],
                round(a["B"]["answer_coverage"]["score"] - a["A"]["answer_coverage"]["score"], 4),
                round(a["C"]["answer_coverage"]["score"] - a["A"]["answer_coverage"]["score"], 4),
                int(a["B"]["fidelity"]["same"]), int(a["C"].get("noop", False)),
                a["A"]["recovered_coverage"]["score"], a["B"]["recovered_coverage"]["score"],
                a["C"]["recovered_coverage"]["score"],
                a["A"]["answer_tokens"]["out"], a["B"]["answer_tokens"]["out"],
                a["C"]["answer_tokens"]["out"],
                a["A"]["geometry"]["cos_intent_recovered"],
                a["B"]["geometry"]["cos_intent_recovered"],
                a["C"]["geometry"]["cos_intent_recovered"],
                a["B"]["geometry"]["overlap_vs_A"], a["C"]["geometry"]["overlap_vs_A"],
            ])

    report = "\n".join(lines)
    (RESULTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print(f"\nWritten: {RESULTS_DIR / 'REPORT.md'} and summary.json")


if __name__ == "__main__":
    main()
