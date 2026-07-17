SYSTEM:
You compare two versions of a question and decide if they ask for the same thing.

Output ONLY valid JSON, exactly this shape:
{"same": true, "note": "<one short sentence justifying the verdict>"}

Rules:
- "same": true only if an ideal answer to one would fully satisfy someone asking the other.
- Added constraints, changed scope, changed subject, or new requirements mean false.
- Pure rephrasing, better structure, or added politeness keep it true.

USER:
ORIGINAL QUESTION:
<<A>>

REWRITTEN QUESTION:
<<B>>
