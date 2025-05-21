# Research Copilot - Technical Stack and Pipeline Design

_**Hypothesis Generation → Literature Study** → Experimentation (Code Generation & Execution) → Reporting (Paper Generation & Review)_

---

## 0 . Executive Summary

|Layer|Purpose|Key Libraries & Services|
|---|---|---|
|**LLM / VLM**|Text & vision generation / analysis|OpenAI gpt‑4o-mini & o4‑mini (via **LiteLLM**)|
|**Prompt & I/O Safety**|Concise prompts, typed outputs|**BAML** (templates + tests)|
|**Agent Class**|Primary interface for interacting with LLMs|**PydanticAI** (Agents)|
|**Workflow Orchestration**|Multi‑agent graphs, retries, state‑sharing|**LangGraph**|
|**Execution Sandboxes**|Safe runtime for generated code|**E2B** (cloud VM) or local Docker|
|**Observability**|Full trace, metrics, feedback|**LangFuse** callbacks & OpenTelemetry|
|**UI**|Researcher interaction|**Gradio**|
|**Persistence**|Versioned artifacts|File store → (plug‑in DB later)|

Stages are chained **left‑to‑right** with explicit artifacts that become first‑class inputs to the next stage:

```
Stage 1  ──►  Stage 2  ──►  Stage 3  ──►  Stage 4
Hypothesis     Citations     Code &      Draft PDF
+ KG           + BibTeX      Results     + Review
```

---

## 1 . Stage 1 – Hypothesis Generation

### Objectives

* Transform a free‑form Research Question into*

1. **Hypothesis** (natural language)
    
2. **Supporting KG** (triples → s/p/o table → interactive graph)
    

### Data Models

```python
class ResearchQuestion(BaseModel):
    question: str

class Triple(BaseModel):
    subject: str; predicate: str; object: str

class HypothesisOutput(BaseModel):
    hypothesis: str
    knowledge_graph: list[Triple]
```

### LangGraph Workflow

```
[Input Q] ──► [HypothesisAgent] ──► [Build PyVis] ──► out
```

* `HypothesisAgent` = BAML prompt → LiteLLM call → validated by PydanticAI  
* Graph HTML & JSON stored in `outputs/<run‑id>/`

### UI Snippet

```python
question = gr.Textbox(...)
model   = gr.Dropdown(["o4‑mini", "o3", "gpt‑4o‑mini", "gpt-4o", "o3-mini"])
run     = gr.Button("Generate")

run.click(fn=run_stage1, inputs=[question, model],
          outputs=[hypothesis_md, triples_json, graph_html])
```

---

## 2 . Stage 2 – Literature Study

### Objectives

_Retrieve N (user‑set) papers, build a **Citation Network** (CN), visualize & persist._

### Data Models

```python
class Paper(BaseModel):
    paper_id: str; title: str; authors: list[str]
    year: int|None; abstract: str|None; references: list[str]

class CitationEdge(BaseModel):
    source_id: str; target_id: str; relation: str="cites"

class CitationNetwork(BaseModel):
    papers: list[Paper]; edges: list[CitationEdge]
```

### Workflow Highlights

1. **Query Builder** → keywords from Stage 1 hypothesis / KG
    
2. **Semantic Scholar & arXiv search** (parallel `ToolNode`s)
    
3. **Merge & Batch‑detail fetch** (refs included)
    
4. **Build Network** (`networkx` → PyVis HTML + JSON)
    
5. **Optional LLM summarizer** (support/refute labels)
    

### UI Add‑ons

_Slider_ to pick paper‑count; interactive CN graph rendered in Gradio HTML; table of seed papers; download JSON.

---

## 3 . Stage 3 – Experimentation (Code Gen + Execution)

### Objectives

Generate Python code for an experiment → run safely → capture **Code + Logs + Plots + Profiling**.

### Core Schemas

```python
class ExperimentTask(BaseModel):
    hypothesis: str
    task_description: str         # user intent
    dataset_reference: str|None

class StaticAnalysisReport(BaseModel):
    passed: bool; issues: list[str]

class ExecutionResult(BaseModel):
    logs: str; status: str; runtime_seconds: float
    memory_mb: float|None
    artifacts: list[str]          # file paths

class ExperimentResult(BaseModel):
    task: ExperimentTask
    code: str
    analysis: StaticAnalysisReport
    exec: ExecutionResult
    success: bool
```

### LangGraph Nodes

```
TaskParse → CodeGen → Lint* → SandboxExec → (retry loop) → PackResult
```

`Lint*` uses Ruff / ty inside a lightweight Docker image.  
`SandboxExec` = E2B Cloud VM or Docker Local VM (`run_code`), 60 s timeout, no outbound net.

### Artifact Folder Example

```
artifacts/exp_2025‑05‑19T18‑04Z/
    experiment.py
    static_analysis.txt
    execution.log
    performance.json
    plot1.png
```

### UI Highlights

_Text box_ for task, _file uploader_ for dataset, real‑time code preview, collapsible console logs, image gallery for plots, download buttons.

---

## 4 . Stage 4 – Reporting (Paper Generation & Review)

### Goal

Draft a complete **LaTeX → PDF** paper and get **VLM feedback** (content + format).

### Section Agents & Graph

```
PrepareCtx
   ├─► Title+Abstract
   ├─► Introduction
   ├─► Related Work
   ├─► Methodology
   ├─► Results
   ├─► Discussion
   └─► Conclusion
          ▼
 AssembleLaTeX → CompilePDF → PDF→Images → GPT‑4V Review → Deliver
```

* Each agent = BAML prompt (+ few‑shot) → GPT‑4o → `PaperSection` object  
* Assemble step inserts BibTeX from Stage 2, figures from Stage 3  
* Compile with `latexmk -pdf -bibtex -interaction=batchmode paper.tex`  
* `pdf2image` → page PNGs; feed each to GPT‑4V with review schema

#### Review Schema

```python
class PageFeedback(BaseModel):
    page: int
    content_issues: list[str]
    format_issues: list[str]

class PaperReview(BaseModel):
    pages: list[PageFeedback]
```

### UI Flow

1. **Input**: extra instructions, author list, template (NeurIPS default)
    
2. **Progress**: live checklist of section completion, compile status
    
3. **Outputs**:
    
    - PDF viewer + download
        
    - Section draft preview
        
    - AI review lists (click page to open image)
        
4. **Iterate**: user edits instructions / sections → regenerate selected parts or whole paper.
    

---

## 5 . Cross‑Stage Orchestration & File‑Tree Scaffold

```
research_copilot/
│
├── stages/
│   ├── stage1_hypothesis/
│   │   ├── models.py         # ResearchQuestion, HypothesisOutput
│   │   ├── prompts/...
│   │   └── pipeline.py       # LangGraph compile()
│   ├── stage2_literature/
│   ├── stage3_experiment/
│   └── stage4_reporting/
│
├── shared/
│   ├── sandbox.py            # E2B / Docker wrapper
│   ├── storage.py            # artifact utils
│   └── logging.py            # LangFuse init
│
├── ui/
│   └── app.py                # Gradio multi‑tab interface
└── main.py                   # load .env, start Gradio
```

---

## 6 . End‑to‑End Pseudocode Skeleton

```python
from stages import stage1, stage2, stage3, stage4

def run_copilot_flow(user_question: str, task_desc: str, dataset_file: str|None):
    # ── Stage 1
    hypo_out = stage1.run(question=user_question)
    # ── Stage 2
    lit_out  = stage2.run(hypo_out, paper_count=15)
    # ── Stage 3
    exp_task = ExperimentTask(
        hypothesis=hypo_out.hypothesis,
        task_description=task_desc,
        dataset_reference=dataset_file,
    )
    exp_res  = stage3.run(exp_task, lit_out)
    # ── Stage 4
    paper    = stage4.run(hypo_out, lit_out, exp_res)
    return paper            # paths to PDF, review JSON, etc.
```

---

## 7 . Future Extension Ideas

|Idea|Quick rationale|
|---|---|
|**Sampling diverse perspectives (Stage 1++)**|Use Quality Diversity to present supporting, neutral, and opposing hypotheses to Research Question.|
|**Idea‑Tree lineage (Stage 2++)**|Use embeddings (SPECTER2) to build evolution graphs.|
|**SWE/MLE coder models (Stage 3++)**|Tree-based code generation, execution, evolutionary improvement loop with multiple specialized models.|
|**Auto‑Revision agent (Stage 4++)**|Feed VLM feedback back to GPT‑4o to patch LaTeX automatically.|
|**Multi‑model routing**|Cheaper models for lint, premium for core writing.|
|**Collaboration mode**|Save drafts, share review links, Git‑like diffs of LaTeX.|
