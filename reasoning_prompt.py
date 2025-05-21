import json, textwrap


def build_reasoning_prompt(user_idea: str, schema: dict, n_triples: int) -> str:
    """
    Returns a single composite prompt that:
    1) walks through Chain‑of‑Thought (ReasonGraph XML tags)
    2) outputs a JSON object that matches *schema* exactly
    """
    schema_str = json.dumps(schema, indent=2)

    return textwrap.dedent(
        f"""
    You are an expert research‑assistant AI.

    —— Part 1 ——
    Think step‑by‑step **in the following format**:

    <thought_process>
      <step number="1">First reflection …</step>
      <step number="2">…</step>
      …
    </thought_process>

    —— Part 2 ——
    When (and only when) you are **completely sure** of the answer,
    output a JSON object that follows **this schema** *exactly*:

    {schema_str}

    Wrap that object in a tag pair `<answer_json>{{ … }}</answer_json>`.

    The subject of all this is:

    \"\"\"{user_idea}\"\"\"

    Remember:
    • provide **exactly {n_triples} triples**
    • do **NOT** add any extra text outside the two tag blocks
    """
    )
