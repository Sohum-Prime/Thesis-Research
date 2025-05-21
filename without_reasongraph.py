"""
Foundation‑Model‑Driven Hypothesis Generator
-------------------------------------------
Gradio demo for HuggingFace Spaces
"""

import json
import os
import traceback

import gradio as gr
import openai
import pandas as pd  # Gradio’s Dataframe prefers pandas
import polars as pl
from pyvis.network import Network
import networkx  # noqa: F401 – ensures networkx is present for PyVis
import html as html_stdlib


# ────────────────────────────────────────────────────────────────────────────────
# Helper (same JSON‑Schema logic you used)
# ────────────────────────────────────────────────────────────────────────────────
def build_schema(n_triples: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "supporting_triples": {
                "type": "array",
                "description": f"Exactly {n_triples} knowledge‑graph triples",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            },
        },
        "required": ["hypothesis", "supporting_triples"],
        "additionalProperties": False,
    }


SYSTEM_TEMPLATE = """
You are an expert research‑assistant AI.
Generate ONE novel, interesting, and useful research *hypothesis*
**and** exactly {n} supporting knowledge‑graph triples.
Return JSON that *strictly* follows the provided schema.
"""


# ────────────────────────────────────────────────────────────────────────────────
# Core generator
# ────────────────────────────────────────────────────────────────────────────────
def generate_hypothesis(
    api_key: str,
    research_idea: str,
    n_triples: int,
) -> tuple[str, pd.DataFrame, str]:
    """
    Returns:
        hypothesis (str),
        triples‑as‑DataFrame (for Gradio Dataframe),
        html (PyVis graph)          ← rendered inside an HTML component
    """
    if not api_key:
        raise gr.Error("Please provide a valid OpenAI API key.")

    # Initialise client each call to avoid key leakage between sessions
    client = openai.OpenAI(api_key=api_key)

    schema = build_schema(n_triples)
    system_prompt = SYSTEM_TEMPLATE.format(n=n_triples)

    try:
        rsp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": research_idea},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "hypothesis_with_triples",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=1500,
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error(f"OpenAI error: {e}")

    try:
        data = json.loads(rsp.output_text)
    except json.JSONDecodeError as e:  # noqa: BLE001
        raise gr.Error(f"Could not parse JSON: {e}")

    hypothesis = data["hypothesis"]
    triples = data["supporting_triples"]

    # Polars → pandas for Gradio
    triples_df = pl.DataFrame(triples).to_pandas()

    # ── Build PyVis graph
    net = Network(height="600px", width="100%", notebook=False, directed=True)
    net.add_node(
        "HYPOTHESIS", label="Hypothesis", title=hypothesis, color="#FFD700", size=28
    )

    nodes_added = {"HYPOTHESIS"}
    for row in triples:
        for entity in (row["subject"], row["object"]):
            if entity not in nodes_added:
                net.add_node(
                    entity, label=entity, title=entity, color="#ADD8E6", size=18
                )
                nodes_added.add(entity)

        net.add_edge(row["subject"], row["object"], label=row["predicate"])

    # Link hypothesis node to each subject
    for row in triples:
        if row["subject"] in nodes_added:
            net.add_edge(
                "HYPOTHESIS", row["subject"], label="relates to", color="orange"
            )

    net.repulsion(node_distance=200, spring_length=200)
    raw_html = net.generate_html("kg.html")  # complete HTML doc
    # html = net.generate_html("kg.html")   # returns full <html> document string

    iframe = f"""
<iframe style="width:100%; height:650px; border:none;"
        srcdoc="{html_stdlib.escape(raw_html)}"></iframe>
"""

    return hypothesis, triples_df, iframe  # html


# ────────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧠 Foundation‑Model‑Driven Hypothesis Generator\n"
        "Generate research hypotheses *and* a supporting knowledge graph, all in one click."
    )

    with gr.Row():
        api_key_in = gr.Textbox(label="🔑 OpenAI API Key", type="password")
        n_triples_in = gr.Number(
            label="✏️ Number of supporting triples", value=10, precision=0
        )
    research_in = gr.Textbox(label="💡 Your research idea / goal", lines=4)
    generate_btn = gr.Button("🚀 Generate", variant="primary")

    gr.Markdown("---")

    hypothesis_out = gr.Textbox(
        label="📜 Generated Hypothesis", lines=3, interactive=False
    )
    triples_out = gr.Dataframe(
        headers=["subject", "predicate", "object"],
        label="📊 Supporting Triples",
        interactive=False,
        wrap=True,
    )
    kg_html_out = gr.HTML(label="🌐 Interactive KG")

    generate_btn.click(
        fn=generate_hypothesis,
        inputs=[api_key_in, research_in, n_triples_in],
        outputs=[hypothesis_out, triples_out, kg_html_out],
    )

# Allow Spaces to set `OPENAI_API_KEY` env var instead of form input if desired
demo.launch(debug=True)
