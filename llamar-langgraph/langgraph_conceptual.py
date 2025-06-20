import marimo

__generated_with = "0.13.15"
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

    # Define the state for our graph.
    class AgentState(TypedDict):
        messages: Annotated[Sequence[AnyMessage], add_messages]
        loop_count: Annotated[int, operator.add] # Reducer to increment
        next_node: str # This key will be used by our conditional edge

    print("Cell 1: Imports and AgentState defined.")#
    return AgentState, END, Literal, START, StateGraph, mo


@app.cell
def _(AgentState):
    # Cell 2: Node Definitions

    # Node 1: Entry point, simulates starting a process
    def entry_node(state: AgentState) -> AgentState:
        print("Executing Entry Node")
        return {
            "messages": [("ai", "Process started by Entry Node.")],
            "loop_count": 0, # Initialize loop_count
        }

    # Node 2: Simulates a tool call or some processing step
    def processing_node_A(state: AgentState) -> AgentState:
        print("Executing Processing Node A")
        return {
            "messages": [("ai", "Data processed by Node A.")],
            "next_node": "path_1" # Example of setting a routing condition
        }

    # Node 3: Another processing step, part of a potential loop
    def loop_node(state: AgentState) -> AgentState:
        print(f"Executing Loop Node (Iteration: {state['loop_count'] + 1})")
        # Increment loop_count using its reducer (operator.add with 1)
        return {
            "messages": [("ai", f"Loop iteration {state['loop_count'] + 1} complete.")],
            "loop_count": 1 # This will be added to the existing loop_count
        }

    # Node 4: A node that decides which path to take next for a conditional edge
    def router_decision_node(state: AgentState) -> AgentState:
        print("Executing Router Decision Node")
        # Simulate a decision, e.g., based on previous messages or some logic
        if len(state["messages"]) % 2 == 0 : # Arbitrary condition for demonstration
            next_destination = "final_node_X"
            message = "Decided to route to Final Node X."
        else:
            next_destination = "final_node_Y"
            message = "Decided to route to Final Node Y."
        return {
            "messages": [("ai", message)],
            "next_node": next_destination
        }

    # Node 5: A final processing node
    def final_node_X(state: AgentState) -> AgentState:
        print("Executing Final Node X")
        return {"messages": [("ai", "Process concluded at Final Node X.")]}

    # Node 6: Another final processing node
    def final_node_Y(state: AgentState) -> AgentState:
        print("Executing Final Node Y")
        return {"messages": [("ai", "Process concluded at Final Node Y.")]}

    # Node 7: A node that could be an alternative start or intermediate step
    def alternative_processing_node(state: AgentState) -> AgentState:
        print("Executing Alternative Processing Node")
        return {
            "messages": [("ai", "Alternative processing complete.")],
            "next_node": "final_node_X" # Example routing
        }


    print("Cell 2: Node functions defined.")
    return (
        alternative_processing_node,
        entry_node,
        final_node_X,
        final_node_Y,
        loop_node,
        processing_node_A,
        router_decision_node,
    )


@app.cell
def _(AgentState, Literal):
    # Cell 3: Edge Logic (Routing Functions)

    # Conditional function for the loop
    def should_continue_loop(state: AgentState) -> Literal["loop_node", "router_decision_node"]:
        print(f"Checking loop condition. Loop count: {state['loop_count']}")
        if state["loop_count"] < 2: # Loop 2 times (0, 1)
            print("Loop continues.")
            return "loop_node"
        else:
            print("Loop finishes. Proceeding to router_decision_node.")
            return "router_decision_node"

    # Conditional function for the main router
    def main_router(state: AgentState) -> Literal["final_node_X", "final_node_Y", "alternative_processing_node"]:
        print(f"Main router deciding based on state['next_node']: {state['next_node']}")
        if state["next_node"] == "final_node_X":
            print("Routing to Final Node X.")
            return "final_node_X"
        elif state["next_node"] == "final_node_Y":
            print("Routing to Final Node Y.")
            return "final_node_Y"
        else:
            # Default or alternative path if next_node isn't X or Y
            print("Routing to Alternative Processing Node as default/fallback.")
            return "alternative_processing_node"


    print("Cell 3: Edge logic functions defined.")
    return main_router, should_continue_loop


@app.cell
def _(
    AgentState,
    END,
    START,
    StateGraph,
    alternative_processing_node,
    entry_node,
    final_node_X,
    final_node_Y,
    loop_node,
    main_router,
    processing_node_A,
    router_decision_node,
    should_continue_loop,
):
    # Cell 4: Graph Construction

    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("entry", entry_node)
    workflow.add_node("processing_A", processing_node_A)
    workflow.add_node("loop_node", loop_node)
    workflow.add_node("router_decision_node", router_decision_node)
    workflow.add_node("final_node_X", final_node_X)
    workflow.add_node("final_node_Y", final_node_Y)
    workflow.add_node("alternative_processing_node", alternative_processing_node)

    # Define the entry point
    workflow.add_edge(START, "entry")

    # Define standard edges
    workflow.add_edge("entry", "processing_A")
    # After processing_A, it will go to the loop_node (or the conditional edge from it)
    workflow.add_edge("processing_A", "loop_node")


    # Define conditional edge for the loop
    # After 'loop_node', the 'should_continue_loop' function is called.
    # Based on its return value ("loop_node" or "router_decision_node"),
    # the graph transitions to the respective node.
    workflow.add_conditional_edges(
        "loop_node",
        should_continue_loop,
        {
            "loop_node": "loop_node", # If "loop_node", it loops back to itself
            "router_decision_node": "router_decision_node" # If "router_decision_node", it moves to the router
        }
    )

    # Define conditional edge for the main router
    # After 'router_decision_node', the 'main_router' function is called.
    workflow.add_conditional_edges(
        "router_decision_node",
        main_router,
        {
            "final_node_X": "final_node_X",
            "final_node_Y": "final_node_Y",
            "alternative_processing_node": "alternative_processing_node"
        }
    )
    workflow.add_edge("alternative_processing_node", "final_node_X") # Example of further connection

    # Define termination points
    workflow.add_edge("final_node_X", END)
    workflow.add_edge("final_node_Y", END)

    print("Cell 4: Graph constructed with nodes and edges.")
    return (workflow,)


@app.cell(column=1)
def _(mo, workflow):
    # Cell 5: Graph Compilation and Visualization

    # Compile the graph
    try:
        graph = workflow.compile()
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

    print("--- Invoking Graph ---")
    initial_state_example = {"messages": [("user", "Hello LangGraph!")], "loop_count": 0, "next_node": ""}
    for s_idx, s_val in enumerate(graph.stream(initial_state_example)):
        print(f"Step {s_idx}: {s_val}")
    print("--- Graph Invocation Complete ---")
    return


if __name__ == "__main__":
    app.run()
