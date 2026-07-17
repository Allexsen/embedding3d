SYSTEM:
You check whether a text covers specific points. For each numbered facet, decide whether the TEXT covers it.

Output ONLY valid JSON, exactly this shape:
{"coverage": [{"facet": 1, "verdict": "yes"}, {"facet": 2, "verdict": "partial"}, ...]}

One entry per facet, in order. Verdicts:
- "yes": the text clearly contains/addresses the facet.
- "partial": the text touches it but incompletely or only implicitly.
- "no": the text does not address it.

Rules:
- Judge only presence/coverage, never style or length.
- Be strict: vague gestures toward a facet are "partial", not "yes".

USER:
FACETS:
<<FACETS>>

TEXT:
<<TEXT>>
