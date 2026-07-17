SYSTEM:
You will see a text that answers some unknown question. Deduce, from the text alone, what its asker apparently wanted to know.

Output ONLY valid JSON, exactly this shape:
{"recovered_intent": "<one detailed paragraph: what the person who prompted this text apparently wanted to know>"}

Rules:
- Be specific and exhaustive: name every distinct thing the asker apparently wanted to know, as reflected by what the text actually covers — subtopics, mechanisms, comparisons, parameters. A rich text implies a rich request; a shallow text implies a vague one.
- Use only the text. Do not evaluate whether the text is good. Do not answer anything yourself.

USER:
TEXT:
<<TEXT>>
