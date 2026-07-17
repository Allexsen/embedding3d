# Analysis freeze — declared before the headline run

Date: 2026-07-14. Config+prompts hash at freeze: see `results/*.json` → `config_hash`
(all headline results must carry the same hash).

- **Cases:** 65 mined cases (20 StackExchange community-labeled "XY problem",
  45 general StackExchange across ux/superuser/workplace/cooking/diy/gardening).
  The 3 pilot cases (`case-000*`) tuned the prompts and are **dev set** —
  excluded from all headline statistics.
- **Primary metric:** answer facet-coverage (0..1), paired deltas **B−A** and
  **C−A**, evaluated with paired bootstrap 95% CI (10k resamples) and paired
  sign-flip permutation p (10k).
- **Pre-declared expectations:** C−A positive and growing with severity;
  B−A ≈ 0 overall (blind rewriting cannot recover unstated intent).
  C's no-op gate should fire mostly on low-severity cases.
- **Secondary channels** (supporting, not headline): recovered-intent coverage
  and cosine, retrieval geometry (source mix, overlap vs A), B fidelity rate,
  token accounting.
- Prompts, config, roles are **frozen** for this run. Any change after this
  point restarts the freeze with fresh cases.
