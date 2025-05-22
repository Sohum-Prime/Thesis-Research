"""
AI Research Copilot: Stage 1 (Hypothesis Gen) + Stage 2 (Literature Study)
-------------------------------------------------------------------------
Gradio demo for HuggingFace Spaces
"""

import json
import os
import traceback
import tempfile
import time
from io import StringIO

import gradio as gr
import openai
import pandas as pd
import polars as pl
from pyvis.network import Network
import networkx  # noqa: F401 – ensures networkx is present for PyVis
import html as html_stdlib
import requests  # For Semantic Scholar API

# ────────────────────────────────────────────────────────────────────────────────
# Stage 1: Foundation-Model-Driven Hypothesis Generator
# (Copied from your provided code, with minor adjustments if needed)
# ────────────────────────────────────────────────────────────────────────────────


# Helper (same JSON-Schema logic you used)
def build_schema(n_triples: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "supporting_triples": {
                "type": "array",
                "description": f"Exactly {n_triples} knowledge-graph triples",
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
You are an expert research-assistant AI.
Generate ONE novel, interesting, and useful research *hypothesis*
**and** exactly {n} supporting knowledge-graph triples.
Return JSON that *strictly* follows the provided schema.
"""


# Core generator for Stage 1
def generate_hypothesis(
    api_key: str,
    research_idea: str,
    n_triples: int,
) -> tuple[str, pd.DataFrame, str, str]:  # Added raw_hypothesis_json for Stage 2
    """
    Returns:
        hypothesis (str),
        triples-as-DataFrame (for Gradio Dataframe),
        html (PyVis graph for KG)
        raw_json_string_from_llm (str) for downstream tasks
    """
    if not api_key:
        raise gr.Error("Please provide a valid OpenAI API key.")

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
        raw_json_output = rsp.output_text
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"OpenAI error: {e}")

    try:
        data = json.loads(raw_json_output)
    except json.JSONDecodeError as e:
        raise gr.Error(
            f"Could not parse JSON from LLM: {e}\nRaw output: {raw_json_output}"
        )

    hypothesis = data["hypothesis"]
    triples = data["supporting_triples"]
    triples_df = pl.DataFrame(triples).to_pandas()

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

    for row in triples:
        if row["subject"] in nodes_added:
            net.add_edge(
                "HYPOTHESIS", row["subject"], label="relates to", color="orange"
            )

    net.repulsion(node_distance=200, spring_length=200)
    raw_html_kg = net.generate_html("kg.html")
    iframe_kg = f"""
<iframe style="width:100%; height:650px; border:none;"
        srcdoc="{html_stdlib.escape(raw_html_kg)}"></iframe>
"""
    return hypothesis, triples_df, iframe_kg, raw_json_output


# ────────────────────────────────────────────────────────────────────────────────
# Stage 2: Literature Study
# ────────────────────────────────────────────────────────────────────────────────

# Data Models (as Python dicts for simplicity, matching spec)
# Paper: { "paper_id": str, "title": str, "authors": list[str],
#          "year": int|None, "abstract": str|None, "references": list[str] }
# CitationEdge: { "source_id": str, "target_id": str, "relation": "cites" }
# CitationNetwork: { "papers": list[Paper], "edges": list[CitationEdge] }

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_REQUEST_TIMEOUT = 20  # seconds


import re
import string  # For punctuation


def extract_keywords_for_s2(
    hypothesis: str,
    kg_triples_df: pd.DataFrame,
    max_hyp_keywords: int = 4,
    max_kg_keywords: int = 3,
    min_word_len: int = 3,
) -> str:
    """
    Extracts a concise set of keywords from hypothesis and KG for Semantic Scholar query.
    V4: Uses string.punctuation for cleaning, keeps internal hyphens.
    """
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "at",
        "from",
        "about",
        "as",
        "into",
        "through",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "just",
        "don",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "ma",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
        "it",
        "its",
        "it's",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "them",
        "their",
        "my",
        "your",
        "his",
        "her",
        "our",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "com",
        "edu",
        "gov",
        "org",
        "net",
        "www",
        "http",
        "https",
        "also",
        "however",
        "therefore",
        "thus",
        "hence",
        "indeed",
        "example",
        "viz",
        "ie",
        "eg",
        "fig",
        "figure",
        "table",
        "sections",
        "section",
        "chapter",
        "appendix",
        "et",
        "al",
        "etc",
        "based",
        "using",
        "via",
        "due",
        "effect",
        "effects",
        "results",
        "impact",
        "show",
        "shows",
        "study",
        "studies",
        "paper",
        "research",
        "article",
        "method",
        "methods",
        "approach",
        "analysis",
        "data",
        "model",
        "models",
        "system",
        "systems",
        "propose",
        "present",
        "investigate",
    }

    # Punctuation to remove from start/end of words. Keeps internal hyphens/apostrophes.
    # string.punctuation is '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # We specifically exclude '-' and "'" from removal if they are internal.
    # word.strip() will handle this better.

    punctuation_to_strip = "".join(c for c in string.punctuation if c not in "-'")

    def clean_and_validate_term(term: str) -> str | None:
        """Lowercase, strip surrounding punctuation, check stopwords, length, and content."""
        cleaned = term.lower().strip(punctuation_to_strip)

        # If after stripping, it's empty, a stopword, or too short, reject.
        if not cleaned or cleaned in stopwords or len(cleaned) < min_word_len:
            return None

        # Ensure it contains at least one alphabetic character (e.g., not just "-.-" or "123")
        # This also helps filter out purely numeric "words" if undesired.
        if not re.search(r"[a-zA-Z]", cleaned):
            return None

        return cleaned

    def get_candidate_terms_from_text(text: str) -> list[str]:
        """Splits text, cleans each token, and returns unique, valid terms in order of appearance."""
        # Split by whitespace and then clean each part.
        # This is preferred over splitting by punctuation to preserve hyphenated terms.
        raw_tokens = re.split(r"\s+", text)  # Split by any whitespace

        valid_terms = []
        for token in raw_tokens:
            if not token:
                continue  # Skip empty strings that might result from multiple spaces
            cleaned = clean_and_validate_term(token)
            if cleaned:
                valid_terms.append(cleaned)
        return list(dict.fromkeys(valid_terms))  # Unique, order-preserved

    # 1. Process Hypothesis
    hypothesis_candidate_terms = get_candidate_terms_from_text(hypothesis)
    selected_hypothesis_keywords = hypothesis_candidate_terms[:max_hyp_keywords]

    # 2. Process KG Entities
    kg_candidate_terms_all = []
    if not kg_triples_df.empty:
        entity_phrases = []
        if "subject" in kg_triples_df.columns:
            # Ensure all are strings before processing
            entity_phrases.extend(str(s) for s in kg_triples_df["subject"].tolist())
        if "object" in kg_triples_df.columns:
            entity_phrases.extend(str(o) for o in kg_triples_df["object"].tolist())

        unique_entity_phrases = list(
            dict.fromkeys(entity_phrases)
        )  # Process unique entity phrases
        for phrase in unique_entity_phrases:
            kg_candidate_terms_all.extend(get_candidate_terms_from_text(phrase))

    # Get unique KG terms that are not already in selected_hypothesis_keywords
    unique_kg_terms_for_selection = []
    # Start with terms already picked from hypothesis to ensure KG adds *new* terms
    seen_terms = set(selected_hypothesis_keywords)
    for term in dict.fromkeys(
        kg_candidate_terms_all
    ):  # Get unique terms from all KG processing
        if term not in seen_terms:
            unique_kg_terms_for_selection.append(term)
            seen_terms.add(
                term
            )  # Add to seen so we don't re-add it if it appears multiple times in KG

    selected_kg_keywords = unique_kg_terms_for_selection[:max_kg_keywords]

    # 3. Combine and Finalize Query
    final_keywords = selected_hypothesis_keywords + selected_kg_keywords

    # Fallback logic if no keywords are generated
    if not final_keywords:
        if (
            hypothesis_candidate_terms
        ):  # If hypothesis processing yielded something, use more of it
            query_string = " ".join(
                hypothesis_candidate_terms[: max_hyp_keywords + max_kg_keywords]
            )
        elif hypothesis:  # Ultimate fallback: first few "raw-ish" words of hypothesis
            # Attempt a very basic cleanup of the raw hypothesis
            raw_hyp_tokens = [
                w.lower().strip(string.punctuation)
                for w in re.split(r"\s+", hypothesis)
            ]
            raw_hyp_tokens_filtered = [
                t
                for t in raw_hyp_tokens
                if t
                and len(t) >= min_word_len
                and t not in stopwords
                and re.search(r"[a-zA-Z]", t)
            ]
            query_string = " ".join(raw_hyp_tokens_filtered[:5])  # Take up to 5
        else:
            query_string = ""  # No hypothesis, no keywords
    else:
        query_string = " ".join(final_keywords)

    # Ensure the query string is not excessively long, even after fallbacks.
    # Semantic Scholar queries are typically best with < 10-15 meaningful terms.
    # Our max_hyp_keywords + max_kg_keywords (e.g., 4+3=7) should keep it concise.
    # If a fallback produced a long string, we might truncate it.
    final_query_tokens = query_string.split()
    if len(final_query_tokens) > (max_hyp_keywords + max_kg_keywords + 2):  # e.g. > 9
        query_string = " ".join(
            final_query_tokens[: (max_hyp_keywords + max_kg_keywords + 2)]
        )

    print(
        f"Revised S2 Query (max_hyp={max_hyp_keywords}, max_kg={max_kg_keywords}): '{query_string}'"
    )
    return query_string.strip()


def search_semantic_scholar(query: str, limit: int) -> list[dict]:
    """Searches Semantic Scholar for papers with rate limit awareness."""
    papers_found = []
    gr.Info(f"Waiting briefly before S2 search call...")
    time.sleep(1)  # Simple delay to help with rate limiting

    try:
        response = requests.get(
            f"{S2_API_BASE}/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "paperId,title,authors.name,year,abstract",
            },
            timeout=S2_REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            gr.Error(
                "Semantic Scholar API rate limit hit (429). Please wait a few minutes and try again."
            )
            return []
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        raw_papers = data.get("data", [])

        for p_raw in raw_papers:
            authors = [
                auth["name"] for auth in p_raw.get("authors", []) if auth.get("name")
            ]
            papers_found.append(
                {
                    "paper_id": p_raw.get("paperId"),
                    "title": p_raw.get("title"),
                    "authors": authors,
                    "year": p_raw.get("year"),
                    "abstract": p_raw.get("abstract"),
                    "references": [],
                }
            )
        gr.Info(f"Found {len(papers_found)} initial papers from S2 search.")
        return papers_found
    except requests.exceptions.HTTPError as http_err:
        # Catch other HTTP errors if needed, but 429 is handled above
        gr.Warning(f"Semantic Scholar search HTTP error: {http_err}")
        return []
    except requests.exceptions.RequestException as e:
        gr.Warning(f"Semantic Scholar search API request error: {e}")
        return []
    except json.JSONDecodeError as e:
        gr.Warning(f"Error decoding S2 search API response: {e}")
        return []


def fetch_paper_details_with_references(paper_ids: list[str]) -> list[dict]:
    """Fetches detailed paper information including references with rate limit awareness."""
    detailed_papers = {}
    if not paper_ids:
        return []

    gr.Info(f"Waiting briefly before S2 batch details call...")
    time.sleep(1.5)  # Slightly longer delay for potentially heavier batch call

    try:
        response = requests.post(
            f"{S2_API_BASE}/paper/batch",
            params={
                "fields": "paperId,title,authors.name,year,abstract,references.paperId"
            },
            json={"ids": paper_ids},
            timeout=S2_REQUEST_TIMEOUT * 2,
        )
        if response.status_code == 429:
            gr.Error(
                "Semantic Scholar API rate limit hit (429) during details fetch. Please wait a few minutes and try again."
            )
            return (
                []
            )  # Or return partially fetched if possible, but for simplicity, fail
        response.raise_for_status()

        s2_detailed_data = response.json()

        for p_detail_raw in s2_detailed_data:
            if p_detail_raw is None:
                continue
            paper_id = p_detail_raw.get("paperId")
            if not paper_id:
                continue

            authors = [
                auth["name"]
                for auth in p_detail_raw.get("authors", [])
                if auth.get("name")
            ]
            references = [
                ref["paperId"]
                for ref in p_detail_raw.get("references", [])
                if ref and ref.get("paperId")
            ]

            detailed_papers[paper_id] = {
                "paper_id": paper_id,
                "title": p_detail_raw.get("title"),
                "authors": authors,
                "year": p_detail_raw.get("year"),
                "abstract": p_detail_raw.get("abstract"),
                "references": references,
            }
        gr.Info(f"Fetched details for {len(detailed_papers)} papers from S2 batch.")
        return list(detailed_papers.values())
    except requests.exceptions.HTTPError as http_err:
        gr.Warning(f"Semantic Scholar batch paper details HTTP error: {http_err}")
        return []
    except requests.exceptions.RequestException as e:
        gr.Warning(f"Semantic Scholar batch paper details API request error: {e}")
        return []
    except json.JSONDecodeError as e:
        gr.Warning(f"Error decoding S2 batch API response: {e}")
        return []


def build_citation_network_object(papers_detailed: list[dict]) -> dict:
    """Builds the CitationNetwork object from a list of detailed papers."""
    citation_network = {"papers": [], "edges": []}
    if not papers_detailed:
        return citation_network

    # Create a set of all fetched paper IDs for quick lookups
    fetched_paper_ids = {p["paper_id"] for p in papers_detailed if p["paper_id"]}

    for paper_data in papers_detailed:
        if not paper_data or not paper_data.get("paper_id"):
            continue

        # Add paper to the network
        # We store the full paper dict in the network's 'papers' list
        # No need to re-create Paper objects if they are already in the correct dict format
        citation_network["papers"].append(paper_data)

        # Add citation edges
        source_id = paper_data["paper_id"]
        for target_id_ref in paper_data.get("references", []):
            if (
                target_id_ref in fetched_paper_ids
            ):  # Only add edge if target is also in our fetched set
                citation_network["edges"].append(
                    {
                        "source_id": source_id,
                        "target_id": target_id_ref,
                        "relation": "cites",
                    }
                )
    return citation_network


def visualize_citation_network(cn_object: dict) -> str:
    """Generates an HTML visualization of the citation network using PyVis."""
    net = Network(
        height="700px",
        width="100%",
        notebook=False,
        directed=True,
        cdn_resources="remote",
    )
    net.repulsion(node_distance=250, spring_length=200)

    if not cn_object or not cn_object.get("papers"):
        return "<p>No citation network data to visualize.</p>"

    for paper in cn_object["papers"]:
        title = paper.get("title", "N/A")
        year = paper.get("year", "N/A")
        label = f"{title[:50]}... ({year})" if title else paper["paper_id"]
        hover_title = f"ID: {paper['paper_id']}\nTitle: {title}\nAuthors: {', '.join(paper.get('authors',[]))}\nYear: {year}"
        net.add_node(
            paper["paper_id"], label=label, title=hover_title, color="#1E90FF", size=15
        )  # DodgerBlue

    for edge in cn_object["edges"]:
        net.add_edge(edge["source_id"], edge["target_id"], title=edge["relation"])

    # Save to a temporary HTML file to get the raw HTML content
    temp_html_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w", encoding="utf-8"
    )
    net.save_graph(temp_html_file.name)
    temp_html_file.close()

    with open(temp_html_file.name, "r", encoding="utf-8") as f:
        raw_html_cn = f.read()
    os.remove(temp_html_file.name)  # Clean up temp file

    iframe_cn = f"""
<iframe style="width:100%; height:750px; border:1px solid #ccc;"
        srcdoc="{html_stdlib.escape(raw_html_cn)}"></iframe>
"""
    return iframe_cn


def run_literature_study(
    hypothesis_text: str,
    kg_triples_df_json: str,  # Pass as JSON string from gr.State or gr.Textbox
    num_papers_to_fetch: int,
) -> tuple[pd.DataFrame, str, str | None, str]:
    """
    Main function for Stage 2.
    Returns:
        papers_table_df (pd.DataFrame for gr.DataFrame),
        cn_viz_html (str for gr.HTML),
        cn_json_filepath (str path for gr.File download or None),
        status_message (str for gr.Markdown)
    """
    if (
        not hypothesis_text or not kg_triples_df_json or kg_triples_df_json == "{}"
    ):  # Check if empty json string
        return (
            pd.DataFrame(),
            "<p>Please generate a hypothesis and KG in Stage 1 first.</p>",
            None,
            "Error: Stage 1 outputs missing.",
        )

    try:
        # Address the FutureWarning for pd.read_json
        kg_triples_df = pd.read_json(StringIO(kg_triples_df_json), orient="split")
    except Exception as e:
        return (
            pd.DataFrame(),
            f"<p>Error parsing KG triples: {e}</p>",
            None,
            f"Error: Could not parse KG data. {e}",
        )

    # 1. Query Builder
    s2_query = extract_keywords_for_s2(
        hypothesis_text, kg_triples_df
    )  # This uses your good version
    if not s2_query:
        return (
            pd.DataFrame(),
            "<p>Could not generate a search query from hypothesis/KG.</p>",
            None,
            "Warning: No keywords for search.",
        )

    # 2. Semantic Scholar search
    initial_papers = search_semantic_scholar(
        s2_query, limit=int(num_papers_to_fetch * 1.5)
    )

    if (
        not initial_papers
    ):  # This will now also be true if search_semantic_scholar returned [] due to 429
        # The error/warning message about 429 or no papers would have been shown by search_semantic_scholar
        return (
            pd.DataFrame(),
            "<p>No papers found or API issue encountered. Check console/notifications.</p>",
            None,
            "Info: Paper search did not yield results or faced an issue.",
        )

    # 3. Batch-detail fetch
    paper_ids_to_fetch_details = [
        p["paper_id"] for p in initial_papers[:num_papers_to_fetch] if p["paper_id"]
    ]

    # Only proceed if there are IDs
    if not paper_ids_to_fetch_details:
        gr.Info("No valid paper IDs to fetch details for after initial search.")
        # Create a basic DataFrame from initial_papers if any, even without details
        papers_for_df_fallback = []
        for p_dict in initial_papers[:num_papers_to_fetch]:
            papers_for_df_fallback.append(
                {
                    "paper_id": p_dict.get("paper_id", "N/A"),
                    "title": p_dict.get("title", "N/A"),
                    "authors": ", ".join(
                        p_dict.get("authors", []) if p_dict.get("authors") else ["N/A"]
                    ),
                    "year": p_dict.get("year", "N/A"),
                }
            )
        papers_table_df_fallback = pd.DataFrame(papers_for_df_fallback)
        return (
            papers_table_df_fallback,
            "<p>No paper details to fetch. Displaying initial search results.</p>",
            None,
            "Info: No further details fetched.",
        )

    papers_with_details = fetch_paper_details_with_references(
        paper_ids_to_fetch_details
    )

    if not papers_with_details:
        # If fetch_paper_details_with_references returned [] (e.g., due to 429 or other error)
        # Fallback to initial papers if detail fetch fails
        gr.Warning(
            "Could not fetch detailed paper information. Using initial search data for the table."
        )
        papers_for_df = []
        for p_dict in initial_papers[
            :num_papers_to_fetch
        ]:  # Use the ones we intended to get details for
            if p_dict["paper_id"] in paper_ids_to_fetch_details:
                papers_for_df.append(
                    {
                        "paper_id": p_dict.get("paper_id", "N/A"),
                        "title": p_dict.get("title", "N/A"),
                        "authors": ", ".join(
                            p_dict.get("authors", [])
                            if p_dict.get("authors")
                            else ["N/A"]
                        ),
                        "year": p_dict.get("year", "N/A"),
                    }
                )
        papers_table_df = pd.DataFrame(papers_for_df)
        # Build citation network with what we have (likely no edges if refs are missing)
        citation_network_obj = build_citation_network_object(
            initial_papers[:num_papers_to_fetch]
        )  # Pass initial papers
        cn_viz_html = visualize_citation_network(citation_network_obj)
        # JSON for download
        cn_json_filepath = None
        if citation_network_obj["papers"]:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".json", encoding="utf-8"
                ) as tmp_json:
                    json.dump(citation_network_obj, tmp_json, indent=2)
                    cn_json_filepath = tmp_json.name
            except Exception as e:
                gr.Warning(f"Could not create JSON file for download: {e}")

        return (
            papers_table_df,
            cn_viz_html,
            cn_json_filepath,
            "Warning: Displaying basic info as detailed fetch failed (possibly rate limit).",
        )

    # 4. Build Network (using papers_with_details)
    citation_network_obj = build_citation_network_object(papers_with_details)

    # 5. Prepare outputs for Gradio
    papers_for_df = []
    for p_dict in citation_network_obj[
        "papers"
    ]:  # Should primarily be from papers_with_details
        papers_for_df.append(
            {
                "paper_id": p_dict.get("paper_id", "N/A"),
                "title": p_dict.get("title", "N/A"),
                "authors": ", ".join(
                    p_dict.get("authors", []) if p_dict.get("authors") else ["N/A"]
                ),
                "year": p_dict.get("year", "N/A"),
            }
        )
    papers_table_df = pd.DataFrame(papers_for_df)

    cn_viz_html = visualize_citation_network(citation_network_obj)
    cn_json_filepath = None
    if citation_network_obj["papers"]:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json", encoding="utf-8"
            ) as tmp_json:
                json.dump(citation_network_obj, tmp_json, indent=2)
                cn_json_filepath = tmp_json.name
        except Exception as e:
            gr.Warning(f"Could not create JSON file for download: {e}")

    status_message = f"Successfully processed {len(citation_network_obj['papers'])} papers and {len(citation_network_obj['edges'])} citation links."
    return papers_table_df, cn_viz_html, cn_json_filepath, status_message


# ────────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky
    )
) as demo:
    gr.Markdown(
        "# 🧠 AI Research Copilot\n"
        "**Stage 1:** Generate research hypotheses & KGs. **Stage 2:** Conduct literature study."
    )

    # Store Stage 1 outputs for Stage 2
    # Use JSON strings for DataFrames to pass them easily
    s1_hypothesis_state = gr.State("")
    s1_triples_df_json_state = gr.State("")  # Store DataFrame as JSON string

    with gr.Tabs():
        with gr.TabItem("🚀 Stage 1: Hypothesis Generation"):
            gr.Markdown(
                "Generate research hypotheses *and* a supporting knowledge graph, all in one click."
            )
            with gr.Row():
                api_key_in = gr.Textbox(
                    label="🔑 OpenAI API Key",
                    type="password",
                    value=os.environ.get(
                        "OPENAI_API_KEY", ""
                    ),  # Pre-fill if env var is set
                )
                n_triples_in = gr.Number(
                    label="✏️ Number of supporting triples",
                    value=5,
                    minimum=3,
                    maximum=20,
                    step=1,
                    precision=0,
                )
            research_in = gr.Textbox(
                label="💡 Your research idea / goal",
                lines=3,
                placeholder="e.g., 'The impact of large language models on scientific discovery'",
            )
            generate_s1_btn = gr.Button(
                "✨ Generate Hypothesis & KG", variant="primary"
            )

            gr.Markdown("---")
            s1_status_out = gr.Markdown("")  # For messages from Stage 1

            hypothesis_out = gr.Textbox(
                label="📜 Generated Hypothesis", lines=2, interactive=False
            )
            triples_out = gr.Dataframe(
                headers=["subject", "predicate", "object"],
                label="📊 Supporting Triples",
                interactive=False,
                wrap=True,
            )
            kg_html_out = gr.HTML(label="🌐 Interactive KG")

            # Hidden components to store raw JSON from Stage 1 for Stage 2
            s1_raw_json_output_for_s2 = gr.Textbox(label="S1 Raw JSON", visible=False)

        with gr.TabItem("📚 Stage 2: Literature Study"):
            gr.Markdown(
                "Use the hypothesis and KG from Stage 1 to find relevant papers and build a citation network."
            )
            num_papers_slider = gr.Slider(
                label="🔎 Number of papers to retrieve (N)",
                minimum=5,
                maximum=30,
                step=1,
                value=10,
            )
            conduct_s2_btn = gr.Button("🔍 Conduct Literature Study", variant="primary")

            s2_status_out = gr.Markdown("")  # For status messages from Stage 2

            gr.Markdown("---")
            retrieved_papers_table_out = gr.DataFrame(
                headers=["paper_id", "title", "authors", "year"],
                label="📚 Retrieved Papers",
                interactive=False,
                wrap=True,
            )
            citation_network_html_out = gr.HTML(
                label="🕸️ Citation Network Visualization"
            )
            citation_network_json_download_out = gr.File(
                label="📄 Download Citation Network (JSON)"
            )

    # Stage 1 Button Click
    def s1_process_and_store(api_key, research_idea, n_triples):
        try:
            hyp, triples_df, kg_html, raw_json = generate_hypothesis(
                api_key, research_idea, n_triples
            )
            # Store for Stage 2
            triples_df_json = (
                triples_df.to_json(orient="split", date_format="iso")
                if not triples_df.empty
                else "{}"
            )
            status_msg = (
                "✅ Stage 1 complete! Hypothesis and KG generated. Ready for Stage 2."
            )
            return hyp, triples_df, kg_html, raw_json, hyp, triples_df_json, status_msg
        except gr.Error as e:  # Catch Gradio errors from generate_hypothesis
            return (
                "",
                pd.DataFrame(),
                f"<p style='color:red;'>Error: {e}</p>",
                "",
                "",
                "",
                str(e),
            )
        except Exception as e:  # Catch any other unexpected errors
            tb_str = traceback.format_exc()
            print(f"Unexpected S1 Error: {e}\n{tb_str}")
            return (
                "",
                pd.DataFrame(),
                f"<p style='color:red;'>An unexpected error occurred: {e}</p>",
                "",
                "",
                "",
                f"Unexpected error: {e}",
            )

    generate_s1_btn.click(
        fn=s1_process_and_store,
        inputs=[api_key_in, research_in, n_triples_in],
        outputs=[
            hypothesis_out,
            triples_out,
            kg_html_out,
            s1_raw_json_output_for_s2,  # Keep passing this for now, might be useful
            s1_hypothesis_state,  # Store for Stage 2
            s1_triples_df_json_state,  # Store for Stage 2
            s1_status_out,
        ],
    )

    # Stage 2 Button Click
    def s2_process_wrapper(hyp_text_state, triples_json_state, num_papers):
        try:
            # Check if stage 1 outputs are present
            if (
                not hyp_text_state
                or not triples_json_state
                or triples_json_state == "{}"
            ):
                gr.Warning(
                    "Please complete Stage 1 first to generate a hypothesis and KG."
                )
                return (
                    pd.DataFrame(),
                    "<p>Stage 1 output missing. Please run Stage 1.</p>",
                    None,
                    "Error: Stage 1 not completed.",
                )

            table_df, cn_html, cn_json_file, s2_status = run_literature_study(
                hyp_text_state, triples_json_state, num_papers
            )
            return table_df, cn_html, cn_json_file, f"✅ {s2_status}"
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Unexpected S2 Error: {e}\n{tb_str}")
            gr.Error(f"An unexpected error occurred in Stage 2: {e}")
            return (
                pd.DataFrame(),
                f"<p style='color:red;'>An unexpected error occurred: {e}</p>",
                None,
                f"Error: {e}",
            )

    conduct_s2_btn.click(
        fn=s2_process_wrapper,
        inputs=[s1_hypothesis_state, s1_triples_df_json_state, num_papers_slider],
        outputs=[
            retrieved_papers_table_out,
            citation_network_html_out,
            citation_network_json_download_out,
            s2_status_out,
        ],
    )

# Allow Spaces to set `OPENAI_API_KEY` env var instead of form input if desired
# demo.queue() # Add queue for longer running tasks
demo.launch(debug=True)
