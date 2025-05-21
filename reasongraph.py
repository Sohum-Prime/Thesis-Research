"""
Standalone Gradio Demo for ReasonGraph's Chain of Thoughts (CoT)
"""

import gradio as gr
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import requests  # For AnthropicAPI if we were using it, OpenAI uses its own client
from openai import (
    OpenAI as OpenAIClient,
)  # Renamed to avoid conflict if we embed OpenAIAPI class

# Configure logging (optional, but good practice)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Embedded from api_base.py (Simplified for OpenAI only initially) ---
@dataclass
class APIResponse:
    text: str
    raw_response: Any
    usage: Dict[str, int]
    model: str


class APIError(Exception):
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} API Error: {message} (Status: {status_code})")


class BaseAPI:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider_name = "base"

    def generate_response(
        self, prompt: str, max_tokens: int = 1024, prompt_format: Optional[str] = None
    ) -> str:
        raise NotImplementedError

    def _format_prompt(self, question: str, prompt_format: Optional[str] = None) -> str:
        if prompt_format:
            return prompt_format.format(question=question)
        return f"""Please answer the question using the following format, with each step clearly marked:
Question: {question}
Let's solve this step by step:
<step number="1">[First step of reasoning]</step>
<step number="2">[Second step of reasoning]</step>
...
<answer>[Final answer]</answer>"""

    def _handle_error(self, error: Exception, context: str = "") -> None:
        error_msg = f"{self.provider_name} API error in {context}: {str(error)}"
        logger.error(error_msg)
        raise APIError(str(error), self.provider_name)


class OpenAIAPI(BaseAPI):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        super().__init__(api_key, model)
        self.provider_name = "OpenAI"
        try:
            self.client = OpenAIClient(api_key=api_key)
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(
        self, prompt: str, max_tokens: int = 1024, prompt_format: Optional[str] = None
    ) -> str:
        try:
            if prompt_format:
                full_prompt_content = prompt_format.format(question=prompt)
            else:
                full_prompt_content = prompt

            logger.info(f"Sending request to OpenAI API with model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt_content}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            self._handle_error(e, "request or response processing")


class APIFactory:
    _providers = {"openai": {"class": OpenAIAPI, "default_model": "gpt-4o-mini"}}

    @classmethod
    def create_api(
        cls, provider: str, api_key: str, model: Optional[str] = None
    ) -> BaseAPI:
        provider = provider.lower()
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider}")

        provider_info = cls._providers[provider]
        api_class = provider_info["class"]
        model_to_use = model or provider_info["default_model"]

        logger.info(
            f"Creating API instance for provider: {provider}, model: {model_to_use}"
        )
        return api_class(api_key=api_key, model=model_to_use)


# --- Embedded from cot_reasoning.py ---
@dataclass
class CoTStep:
    number: int
    content: str


@dataclass
class CoTResponse:
    question: str
    steps: List[CoTStep]
    answer: Optional[str] = None


@dataclass
class VisualizationConfig:
    max_chars_per_line: int = 40
    max_lines: int = 4
    truncation_suffix: str = "..."


def wrap_text_for_mermaid(text: str, config: VisualizationConfig) -> str:
    """Wrap text to fit within box constraints for Mermaid."""
    text = text.replace("\n", " ").replace('"', "'")
    wrapped_lines = textwrap.wrap(
        text,
        width=config.max_chars_per_line,
        break_long_words=True,
        break_on_hyphens=True,
        max_lines=config.max_lines,  # textwrap can handle max_lines directly
        placeholder=config.truncation_suffix,  # and a placeholder for truncated lines
    )
    # If textwrap didn't use max_lines (e.g. older versions) or if further refinement is needed:
    if len(wrapped_lines) > config.max_lines:
        wrapped_lines = wrapped_lines[: config.max_lines]
        # Ensure the last line correctly shows truncation if it was indeed cut short by max_lines
        # and not just by char_per_line limit within that line.
        # textwrap with placeholder should handle this, but as a fallback:
        if not wrapped_lines[-1].endswith(config.truncation_suffix):
            if len(wrapped_lines[-1]) > config.max_chars_per_line - len(
                config.truncation_suffix
            ):
                wrapped_lines[-1] = (
                    wrapped_lines[-1][
                        : config.max_chars_per_line - len(config.truncation_suffix)
                    ]
                    + config.truncation_suffix
                )
            else:
                wrapped_lines[-1] += config.truncation_suffix

    return "<br>".join(wrapped_lines)


def parse_cot_response(response_text: str, question: str) -> CoTResponse:
    step_pattern = r'<step number="(\d+)">\s*(.*?)\s*</step>'
    steps = []
    for match in re.finditer(step_pattern, response_text, re.DOTALL):
        number = int(match.group(1))
        content = match.group(2).strip()
        steps.append(CoTStep(number=number, content=content))

    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    answer_match = re.search(answer_pattern, response_text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else "No answer tag found"

    steps.sort(key=lambda x: x.number)
    return CoTResponse(question=question, steps=steps, answer=answer)


def create_cot_mermaid_diagram(
    cot_response: CoTResponse, config: VisualizationConfig
) -> str:
    diagram_parts = ["graph TD"]

    question_content = wrap_text_for_mermaid(cot_response.question, config)
    diagram_parts.append(f'    Q["{question_content}"]')
    diagram_parts.append(f"    class Q questionStyle;")

    prev_node_id = "Q"
    if cot_response.steps:
        for i, step in enumerate(cot_response.steps):
            content = wrap_text_for_mermaid(step.content, config)
            node_id = f"S{step.number}"
            diagram_parts.append(f'    {node_id}["Step {step.number}:<br>{content}"]')
            diagram_parts.append(f"    class {node_id} stepStyle;")
            diagram_parts.append(f"    {prev_node_id} --> {node_id}")
            prev_node_id = node_id

    if cot_response.answer:
        answer_content = wrap_text_for_mermaid(cot_response.answer, config)
        diagram_parts.append(f'    A["Answer:<br>{answer_content}"]')
        diagram_parts.append(f"    class A answerStyle;")
        diagram_parts.append(f"    {prev_node_id} --> A")

    # FIX: Removed non-standard CSS attributes (padding, rx, ry) from classDef
    diagram_parts.extend(
        [
            "    classDef questionStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px",
            "    classDef stepStyle fill:#f9f9f9,stroke:#333,stroke-width:2px",
            "    classDef answerStyle fill:#d4edda,stroke:#28a745,stroke-width:2px",
            "    linkStyle default stroke:#666,stroke-width:2px",
        ]
    )

    return f'<div class="mermaid">\n{chr(10).join(diagram_parts)}\n</div>'


# --- CoT Specific Prompt ---
COT_PROMPT_FORMAT = """Please answer the question using the following format by Chain-of-Thoughts, with each step clearly marked:

Question: {question}

Let's solve this step by step:
<step number="1">
[First step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Final answer]
</answer>"""

COT_EXAMPLE_QUESTION = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"


# --- Gradio Function ---
def generate_cot_visualization(
    api_key: str,
    model_name: str,
    question: str,
    max_tokens_val: int,
    chars_per_line_val: int,
    max_lines_val: int,
):
    if not api_key:
        raise gr.Error("Please provide an API key.")
    if not question:
        raise gr.Error("Please enter a question.")

    try:
        api = APIFactory.create_api(
            provider="openai", api_key=api_key, model=model_name
        )
        raw_llm_response = api.generate_response(
            prompt=question,
            max_tokens=int(max_tokens_val),
            prompt_format=COT_PROMPT_FORMAT,
        )
        parsed_cot_response = parse_cot_response(raw_llm_response, question)
        viz_config = VisualizationConfig(
            max_chars_per_line=int(chars_per_line_val), max_lines=int(max_lines_val)
        )
        mermaid_html = create_cot_mermaid_diagram(parsed_cot_response, viz_config)

        mermaid_html = "<div class='mermaid'>\n" + "\n".join(diagram_parts) + "\n</div>"

        return raw_llm_response, mermaid_html

    except APIError as e:
        logger.error(f"API Error: {e}")
        raise gr.Error(f"API Error: {e.message}")
    except Exception as e:
        logger.exception("An unexpected error occurred")
        raise gr.Error(f"An unexpected error occurred: {str(e)}")


# --- Gradio UI ---
mermaid_js = """
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
<script>
  // render everything having class="mermaid" once the page is ready
  window.addEventListener("load", () => mermaid.initialize({startOnLoad:true, securityLevel:'loose'}));
</script>
"""

with gr.Blocks(head=mermaid_js, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ReasonGraph Lite: Chain of Thoughts (CoT) Demo")
    gr.Markdown(
        "Visualizes the reasoning path of an LLM using the Chain of Thoughts method."
    )

    with gr.Row():
        api_key_in = gr.Textbox(
            label="🔑 OpenAI API Key", type="password", placeholder="sk-..."
        )
        model_in = gr.Dropdown(
            label="🧠 OpenAI Model",
            choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"],
            value="gpt-4o-mini",
        )

    question_in = gr.Textbox(
        label="❓ Your Question",
        lines=3,
        placeholder="Enter your question here...",
        value=COT_EXAMPLE_QUESTION,
    )

    with gr.Accordion("⚙️ Advanced Settings", open=False):
        max_tokens_in = gr.Slider(
            label="Max Tokens for LLM", minimum=50, maximum=4000, value=1024, step=50
        )
        with gr.Row():
            chars_per_line_in = gr.Slider(
                label="Chars per Line (Node)", minimum=20, maximum=100, value=40, step=5
            )
            max_lines_in = gr.Slider(
                label="Max Lines (Node)", minimum=1, maximum=10, value=4, step=1
            )

    generate_btn = gr.Button("🚀 Generate ReasonGraph", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## Results")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Raw LLM Output")
            raw_output_out = gr.Textbox(
                lines=15,
                label="Raw Output",
                interactive=False,
                elem_id="raw-output-box",
            )
        with gr.Column(scale=1):
            gr.Markdown("### 🌊 ReasonGraph Visualization")
            mermaid_out = gr.HTML(label="🌊 ReasonGraph")

    generate_btn.click(
        fn=generate_cot_visualization,
        inputs=[
            api_key_in,
            model_in,
            question_in,
            max_tokens_in,
            chars_per_line_in,
            max_lines_in,
        ],
        outputs=[raw_output_out, mermaid_out],
    )

if __name__ == "__main__":
    demo.launch(debug=True)
