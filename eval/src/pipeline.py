"""The experiment pipeline. Each stage is idempotent: it fills its slot in the
case's result dict and skips if already present, so runs resume for free and
the executor can batch stages across cases (one Ollama model resident at a time).

Arms:
  A = the prompt as the user typed it.
  B = auto-rewrite: optimizer sees ONLY the prompt (middleware baseline).
  C = mirror loop: user (who privately knows their intent) sees what their
      prompt conveys and revises — the product being tested.
"""
import numpy as np

import corpus
import embedder
from common import CONFIG, config_hash, fill, load_prompt
from ollama_client import chat, chat_json

VERDICT_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def _facet_block(facets: list[str]) -> str:
    return "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facets))


def _judge_coverage(facets: list[str], text: str) -> dict:
    p = load_prompt("judge_facets")
    out = chat_json("judge", p["system"],
                    fill(p["user"], facets=_facet_block(facets), text=text),
                    ("coverage",))
    entries = out["data"]["coverage"]
    scores = []
    for i in range(len(facets)):
        entry = next((e for e in entries if e.get("facet") == i + 1), None)
        verdict = (entry or {}).get("verdict", "no")
        scores.append(VERDICT_SCORE.get(verdict, 0.0))
    return {"per_facet": scores, "score": round(float(np.mean(scores)), 4),
            "raw": entries}


# ---------------------------------------------------------------- stages

def stage_extract(case, state):
    p = load_prompt("extractor")
    out = chat_json("extractor", p["system"],
                    fill(p["user"], prompt=case["prompt"], evidence=case["evidence"]),
                    ("intent", "facets"))
    state["intent"] = out["data"]["intent"]
    state["facets"] = [str(f) for f in out["data"]["facets"]][:6]


def stage_severity(case, state):
    """How much of the true intent does the prompt itself convey? severity = 1 - conveyed."""
    cov = _judge_coverage(state["facets"], case["prompt"])
    state["prompt_conveys"] = cov
    state["severity"] = round(1.0 - cov["score"], 4)


def stage_mirror(case, state):
    p = load_prompt("mirror")
    out = chat_json("mirror", p["system"], fill(p["user"], prompt=case["prompt"]),
                    ("conveyed_intent",))
    state["mirror"] = {"conveyed_intent": out["data"]["conveyed_intent"],
                       "foreclosed": out["data"].get("foreclosed", ""),
                       "tokens_out": out["meta"]["tokens_out"]}


def stage_rewrite_b(case, state):
    p = load_prompt("optimizer")
    out = chat_json("optimizer", p["system"], fill(p["user"], prompt=case["prompt"]),
                    ("rewritten",))
    state["arms"]["B"]["prompt"] = out["data"]["rewritten"]
    state["arms"]["B"]["overhead_tokens"] = out["meta"]["tokens_out"]


def stage_fidelity_b(case, state):
    p = load_prompt("judge_fidelity")
    out = chat_json("judge", p["system"],
                    fill(p["user"], a=case["prompt"], b=state["arms"]["B"]["prompt"]),
                    ("same",))
    state["arms"]["B"]["fidelity"] = {"same": bool(out["data"]["same"]),
                                      "note": out["data"].get("note", "")}


def stage_revise_c(case, state):
    p = load_prompt("reviser")
    out = chat_json("reviser", p["system"],
                    fill(p["user"], prompt=case["prompt"], intent=state["intent"],
                         conveyed=state["mirror"]["conveyed_intent"]),
                    ("revised",))
    revised = str(out["data"]["revised"]).strip()
    changed = bool(out["data"].get("changed", True)) and revised != case["prompt"].strip()
    # no-op gate: a mirror that rewrites already-good prompts is a bad mirror —
    # when the simulated asker declines to revise, arm C rides the original
    # prompt (and, via the cache, inherits arm A's answer verbatim)
    state["arms"]["C"]["prompt"] = revised if changed else case["prompt"]
    state["arms"]["C"]["noop"] = not changed
    state["arms"]["C"]["overhead_tokens"] = (state["mirror"]["tokens_out"]
                                             + out["meta"]["tokens_out"])


def stage_answers(case, state):
    p = load_prompt("answerer")
    for arm in CONFIG["arms"]:
        slot = state["arms"][arm]
        if "answer" in slot:
            continue
        out = chat("answerer", p["system"], fill(p["user"], prompt=slot["prompt"]),
                   json_mode=False, max_tokens=CONFIG["max_answer_tokens"])
        slot["answer"] = out["content"].strip()
        slot["answer_tokens"] = {"in": out["tokens_in"], "out": out["tokens_out"]}


def stage_deduce(case, state):
    p = load_prompt("deducer")
    for arm in CONFIG["arms"]:
        slot = state["arms"][arm]
        if "recovered_intent" in slot:
            continue
        out = chat_json("deducer", p["system"], fill(p["user"], text=slot["answer"]),
                        ("recovered_intent",))
        slot["recovered_intent"] = out["data"]["recovered_intent"]


def stage_judge_answers(case, state):
    for arm in CONFIG["arms"]:
        slot = state["arms"][arm]
        if "answer_coverage" not in slot:
            slot["answer_coverage"] = _judge_coverage(state["facets"], slot["answer"])


def stage_judge_recovered(case, state):
    for arm in CONFIG["arms"]:
        slot = state["arms"][arm]
        if "recovered_coverage" not in slot:
            slot["recovered_coverage"] = _judge_coverage(state["facets"],
                                                         slot["recovered_intent"])


def stage_geometry(case, state):
    """Local, LLM-free channel: intent-recovery cosine + corpus retrieval per arm."""
    intent_vec = embedder.embed([state["intent"]])[0]
    base_neighbors = None
    for arm in CONFIG["arms"]:
        slot = state["arms"][arm]
        vecs = embedder.embed([slot["prompt"], slot["recovered_intent"]])
        neighbors = corpus.top_k(vecs[0])
        if arm == "A":
            base_neighbors = neighbors
        slot["geometry"] = {
            "cos_intent_recovered": round(embedder.cos(intent_vec, vecs[1]), 4),
            "cos_intent_prompt": round(embedder.cos(intent_vec, vecs[0]), 4),
            "source_mix": corpus.source_mix(neighbors),
            "overlap_vs_A": corpus.overlap(base_neighbors, neighbors),
            "top3": [{"source": n["source"], "text": n["text"][:100]}
                     for n in neighbors[:3]],
        }
    if state["arms"]["B"].get("fidelity") is not None:
        a_vec, b_vec = embedder.embed([case["prompt"], state["arms"]["B"]["prompt"]])
        state["arms"]["B"]["fidelity"]["cos_prompts"] = round(embedder.cos(a_vec, b_vec), 4)


# name -> (fn, model-owner) — order groups same-model stages to minimize VRAM swaps
STAGES = [
    ("extract", stage_extract),
    ("severity", stage_severity),
    ("mirror", stage_mirror),
    ("rewrite_b", stage_rewrite_b),
    ("fidelity_b", stage_fidelity_b),
    ("revise_c", stage_revise_c),
    ("answers", stage_answers),
    ("deduce", stage_deduce),
    ("judge_answers", stage_judge_answers),
    ("judge_recovered", stage_judge_recovered),
    ("geometry", stage_geometry),
]


def new_state(case: dict) -> dict:
    # key order = reading order in the result file: what was typed, what was
    # meant (intent/facets), how bad the gap is — THEN the three arms
    return {
        "case_id": case["id"],
        "config_hash": config_hash(),
        "typed_prompt": case["prompt"],
        "intent": None,
        "facets": None,
        "severity": None,
        "prompt_conveys": None,
        "mirror": None,
        "stages_done": [],
        "arms": {"A": {"prompt": case["prompt"], "overhead_tokens": 0},
                 "B": {}, "C": {}},
    }
