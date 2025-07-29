import marimo

__generated_with = "0.14.12"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    # Cell 1: Imports and State Definition
    import marimo as mo
    from typing import TypedDict, Annotated, List, Optional, Sequence, Literal
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages, AnyMessage # Using AnyMessage for broader compatibility
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
    from IPython.display import Image, display
    import operator
    from baml_client import b 

    # Define the state for our graph.
    class GraphState(TypedDict):
        messages: Annotated[Sequence[AnyMessage], add_messages]
        user_text: str
        summary: str

    print("Cell 1: Imports and GraphState defined.")
    return END, GraphState, START, StateGraph, b, mo


@app.cell
def _(GraphState, b):
    # Cell 2: Node Definitions

    def entry_node(state: GraphState) -> GraphState:
        """Pretend we just received a user message to summarise."""
        txt = state["user_text"]
        return {"messages": [("user", txt)]}

    def summarise_node(state: GraphState) -> GraphState:
        """Call the BAML function inside a LangGraph node."""
        result = b.SummarizeText({"text": state["user_text"]})
        return {
            "messages": [("ai", result.summary)],
            "summary": result.summary,
        }
    
    print("Cell 2: Node functions defined.")
    return entry_node, summarise_node


@app.cell
def _(END, GraphState, START, StateGraph, entry_node, summarise_node):
    # Cell 3: Wire up the Graph

    g = StateGraph(GraphState)
    g.add_node("start", entry_node)
    g.add_node("summarise", summarise_node)

    g.add_edge(START, "start")
    g.add_edge("start", "summarise")
    g.add_edge("summarise", END)


    print("Cell 3: Edge logic functions defined.")
    return (g,)


@app.cell(column=1)
def _(g, mo):
    # Cell 5: Graph Compilation and Visualization

    # Compile the graph
    try:
        graph = g.compile()
        print("Graph compiled successfully.")
    except ImportError:
        print("LangGraph is not installed. Please install it to compile and visualize the graph.")
    except Exception as e:
        print(f"An error occurred during graph compilation: {e}")

    # Visualize the graph
    # This will generate a mermaid diagram of the graph.
    # In Marimo, this should display the image directly if mo.mermaid is working as expected.
    # If not, it might save to a file or require a specific Marimo UI element for images.
    image_data = graph.get_graph().draw_mermaid()
    mo.mermaid(diagram=image_data)
    return (graph,)


@app.cell
def _(graph):
    # Cell 6: Graph Invocation and Testing

    if __name__ == "__main__":
        initial = {"user_text": "LangGraph is an orchestration library that treats \
    agent workflows as data‑flow graphs.  It offers persistence, streaming and \
    human‑in‑the‑loop features out of the box."}
        for step, state in enumerate(graph.stream(initial)):
            print(f"Step {step}: {state}")
    return


if __name__ == "__main__":
    app.run()
