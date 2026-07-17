# eval — the intent-utterance gap, measured (n=65, archived)

**Status: completed & archived (July 2026).** One pre-registered study, fully
reproducible from this directory. The broader research program it was part of
is paused — see [Why this stopped here](#why-this-stopped-here).

## The question

People's prompts under-state what they actually want — they don't monitor what
their words convey, and mid-thought framing bakes the wrong question in (the
classic XY problem: asking "light green or cyan?" when the goal is "the best
accent color for this dashboard"). Does that gap measurably hurt AI answers,
and what actually recovers the loss: blind automatic prompt rewriting, or
showing the asker what their prompt conveys so *they* revise it?

## Design

65 real questions from StackExchange (20 threads the community itself labeled
"XY problem" + 45 general questions across ux/superuser/workplace/cooking/diy/
gardening). Each case is `{prompt, evidence}`: the title is what the person
asked; their own post body is ground-truth evidence of what they meant.
Attribution (CC BY-SA) is stored per case.

Per case, run entirely on local models (Ollama, fixed seeds, one model per
role across four families so nobody judges its own relatives):

1. **Extractor** (llama3.1:8b) reads prompt+evidence → intent + 3–6 checkable
   **facets**. **Severity** = share of facets the raw prompt fails to convey.
2. Three arms produce prompts for the same **answerer** (qwen2.5:7b), which
   never knows which arm it serves:
   - **A** — the prompt as typed;
   - **B** — *blind auto-rewrite*: an optimizer (mistral:7b) that sees only
     the prompt — it cannot know unstated intent, by design;
   - **C** — *mirror loop*: a model (gemma2:9b) states what the prompt
     conveys; a simulated asker holding the true intent revises (with a
     no-op option). This simulates a user seeing a pre-flight "here's what
     your question actually asks" tool.
3. Answers are scored per-facet by a **judge** (llama3.1:8b), blind to arms;
   a **deducer** (gemma2:9b) reads each answer alone and recovers the
   apparent intent (round-trip check); an LLM-free geometric channel embeds
   everything (same mpnet + int8 corpus math as the website) for retrieval
   overlap and register-mix shifts.

The analysis was **pre-registered in [FREEZE.md](FREEZE.md)** before the run:
primary metric (paired facet-coverage deltas B−A and C−A, bootstrap 95% CI,
sign-flip permutation test), expectations, and the dev/headline split. The
three `case-000*` files are the development set that tuned the prompts — they
are excluded from all headline statistics.

## Results (n=65)

| Arm | Mean answer facet-coverage |
|---|---|
| A — raw prompt | 0.550 |
| B — blind auto-rewrite | 0.513 |
| C — mirror loop | **0.626** |

- **C − A = +0.076**, 95% CI [+0.033, +0.120], permutation p = **0.001**.
- **B − A = −0.037**, CI [−0.077, +0.003], p = 0.076 — blind rewriting
  *trends harmful* despite preserving the question's meaning in 65/65 cases
  (judged): you can't reword in what the words never contained.
- On the 20 community-labeled **XY cases** the mirror's effect nearly
  doubles (C−A = +0.115, p = 0.03) while raw prompts cover barely half of
  the asker's intent (0.487) — and blind rewriting is at its most harmful.
- Not a verbosity artifact: answer-length gain correlates *negatively* with
  coverage gain (r = −0.13); C's advantage is larger on shorter answers.
- Raw prompts left ~45% of the askers' demonstrable intent uncovered.
- Pre-registered expectation that failed, reported as required: C's no-op
  gate ("your prompt is fine, send as-is") fired 0/65 — the simulated asker
  always revised. The low-severity stratum still shows ≈0 net effect, but
  the gate itself is unvalidated.

Full aggregates: [results/REPORT.md](results/REPORT.md) ·
per-case table: [results/results.csv](results/results.csv) ·
complete transcripts (every prompt variant, answer, verdict): `results/*.json`.

## Reproducing

```bash
# needs: Ollama with qwen2.5:7b, llama3.1:8b, gemma2:9b, mistral:7b
# and a Python env with numpy/pandas/scipy/requests/sentence-transformers
cd eval
python src/run.py --list     # queue status
python src/run.py            # process pending cases
python src/report.py         # regenerate REPORT.md / summary.json / results.csv
python src/mine.py           # mine fresh StackExchange cases (KB-scale API calls)
```

Every LLM call is seeded, logged (`runs/calls.jsonl`) and content-address
cached, so reruns are deterministic and interrupted runs resume free. Results
embed a hash of config + role prompts; the committed state matches the
archived results' hash (`a46c583fdf9c4384`). `src/audit.py` can re-judge a
seeded sample on any OpenAI-compatible frontier endpoint for an agreement
check — **no such audit was run** (see below).

## Why this stopped here

The study above is the defensible core. The larger program around it was
deliberately dropped, for reasons worth recording:

- **Model quality.** Every judgment here comes from 7–9B local models. They
  needed several pilot iterations to stop leaking framing into the ground
  truth (all documented in git history), and while the design mitigates bias
  (family separation, blind judging, facet checklists, an LLM-free geometric
  channel), the planned frontier audit was never run — the numbers are
  small-models-grading-small-models until someone runs `audit.py`.
- **Hardware.** 12 GB VRAM caps the roles at 7–9B. The local frontier-audit
  option (a disk-streamed 744B MoE) required a ~370 GB download at
  0.05–1 tok/s — declined as disproportionate for a side project.
- **Data quality & a structural flaw.** The planned extension — real
  multi-turn chats (WildChat), compressing each conversation's revealed
  intent into one message — was built and piloted, then dropped: roughly a
  third of real chat data is creative/roleplay noise, only ~28% of
  structurally usable conversations pursue a single under-stated goal, and,
  decisively, the "one message could have replaced the chat" claim is
  partially tautological — users' later turns often *react to answers*
  (information they could not have stated at t=0), so the experiment
  measures the compressibility of transcripts, not achievable user behavior.
- **Scope.** This was a side branch of a side project. The honest stopping
  point is one completed, pre-registered result with published transcripts —
  not a growing pile of half-defended claims.

What survives elsewhere: the website's live demos of the same thesis
(register-driven retrieval divergence, generation divergence), and this
directory as the evidence that the "mirror over ghostwriter" idea holds on
real questions with p = 0.001.
