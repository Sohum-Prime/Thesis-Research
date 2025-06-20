import marimo

__generated_with = "0.13.15"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""### Initialize State for 'Plan-Act-Verify-Correct' framework""")
    return


@app.cell
def _():
    import marimo as mo
    import os, sys, operator, json, re, textwrap
    from typing import TypedDict, Annotated, Sequence, Dict, List
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    class AgentStateSlice(TypedDict, total=False):
        observation:str; state:str; previous_action:str
        previous_failures:str; subtask:str; failure_reason:str

    class LLAMARState(TypedDict):
        # global task context
        task_description: str
        open_subtasks: List[str]        # GO,t  in the paper
        completed_subtasks: List[str]   # GC,t
        memory: str                     # M_t

        # per-agent slices – key = agent name
        agent_states: Dict[str, AgentStateSlice]  # {'observation', 'state', 'prev_action', 'prev_failures', 'subtask', 'failure_reason'}

        # transient conversation plumbing
        plan: List[str]                 # cached plan from Planner
        last_actions: List[str]         # a_t-1
        corrective_action: List[str]    # a_c,t-1
        turn_number: Annotated[int, operator.add]
        route: str                      # used by edge router

    print(f"Current working directory: {os.getcwd()}")
    return END, LLAMARState, StateGraph, json, mo, re


@app.cell(hide_code=True)
def _():
    # from env import SAREnv
    # from sar_logging import SARLogger

    # # --- Configuration for the Environment ---
    # # We'll use these constants to initialize the environment.
    # # These match the defaults from the original llamar.py for consistency.
    # NUM_AGENTS = 2
    # SCENE = 1
    # SEED = 42
    # BASELINE_NAME = "llamar_langgraph"

    # # --- Instantiation ---
    # # 1. Initialize the Search and Rescue Environment.
    # #    This object will manage the simulation state, handle actions, and provide observations.
    # env = SAREnv(num_agents=NUM_AGENTS, scene=SCENE, seed=SEED, save_frames=False)

    # # 2. Initialize the logger.
    # #    This will be used by the ActionExecutionNode later to log results.
    # logger = SARLogger(env=env, baseline_name=BASELINE_NAME)

    # # 3. Reset the environment to get the starting state.
    # #    env.reset() returns the initial input_dict for the LLMs.
    # initial_env_dict = env.reset()

    # # --- Initial State Construction ---
    # # 4. Create the initial state for our LangGraph workflow.
    # #    This dictionary populates the LLAMARState TypedDict with starting values.
    # initial_agent_states = {}
    # for agent_name in env.agent_names:
    #     initial_agent_states[agent_name] = {
    #         "observation": initial_env_dict.get(f"{agent_name}'s observation", ""),
    #         "state": initial_env_dict.get(f"{agent_name}'s state", ""),
    #         "previous_action": initial_env_dict.get(f"{agent_name}'s previous action", ""),
    #         "previous_failures": initial_env_dict.get(f"{agent_name}'s previous failures", "None"),
    #     }

    # initial_state = LLAMARState(
    #     task_description=env.task,
    #     plan=[],
    #     completed_subtasks=[],
    #     agent_states=initial_agent_states,
    #     current_actor_agent_id=env.agent_names[0], # Start with the first agent
    #     turn_number=0,
    #     last_actions=[],
    #     action_success=True, # Assume success at the start
    #     route="", # No route determined yet
    # )

    # print("Cell 2: SAREnv and SARLogger instantiated.")
    # print(f"Task Description: {initial_state['task_description']}")
    # mo.ui.table(initial_state)
    return


@app.cell(hide_code=True)
def _():
    # # marimo cell ────────────────────────────────────────────────────────────────
    # import json, importlib
    # from pathlib import Path

    # from env import SAREnv
    # from sar_logging import SARLogger

    # # 0️⃣  swap the scene module before env.reset() ------------------------------
    # # load the generic initialiser module once
    # generic_scene_mod = importlib.import_module("Scenes.scene_initializer")

    # # monkey-patch get_scene_initializer so that every scene id returns *this*
    # def use_generic(scene_id: int):
    #     return generic_scene_mod, generic_scene_mod
    # import Scenes.get_scene_init as gsi
    # gsi.get_scene_initializer = use_generic
    # import env as env_mod
    # env_mod.get_scene_initializer = use_generic

    # # 1️⃣  repo config ------------------------------------------------------------
    # NUM_AGENTS = json.load(open("multiagent_config.json"))["num_agents"]

    # # 2️⃣  simulator --------------------------------------------------------------
    # env = SAREnv(num_agents=NUM_AGENTS, scene=1, seed=42, save_frames=True)
    # raw = env.reset()                                            # now succeeds

    # # 3️⃣  logger -----------------------------------------------------------------
    # logger = SARLogger(env=env, baseline_name="llamar_langgraph")

    # # 4️⃣  initial LangGraph state ------------------------------------------------
    # state = LLAMARState(
    #     task_description   = env.task,
    #     open_subtasks      = [],
    #     completed_subtasks = [],
    #     memory             = "",
    #     plan               = [],
    #     last_actions       = [],
    #     corrective_action  = [],
    #     turn_number        = 0,
    #     route              = "",
    #     agent_states = {
    #         name: {
    #             "observation"      : raw.get(f"{name}'s observation", ""),
    #             "state"            : raw.get(f"{name}'s state",        ""),
    #             "previous_action"  : raw.get(f"{name}'s previous action", ""),
    #             "previous_failures": raw.get(f"{name}'s previous failures", "None"),
    #         }
    #         for name in env.agent_names
    #     },
    # )

    # print("✅  Environment initialised, logger attached; LLAMARState seeded.")
    # mo.ui.table(state)
    return


@app.cell(hide_code=True)
def _(json, re):
    from env import SAREnv

    def join_conjunction(l, conj):
        if len(l)<2: return str(l[0])
        if len(l)==2: return f"{l[0]} {conj} {l[1]}"
        return ", ".join(l[:-1]) + f", {conj} {l[-1]}"

    def extract_json_block(s:str)->str:
        """Grabs the first ```json … ``` fenced block."""
        m=re.search(r"```json(.*?)```", s, flags=re.S)
        return m.group(1) if m else s

    def convert_dict_to_string(d:dict)->str:
        return json.dumps(d, indent=2)

    def process_action_llm_output(d:dict):
        acts=[v for k,v in d.items() if k.endswith("action")]
        return acts, d.get("reason"), d.get("subtask"), d.get("memory"), d.get("failure reason")

    NUM_AGENTS = json.load(open("multiagent_config.json"))["num_agents"]

    # AGENT_NAMES global variable from env_new contains all the agent names (6 of them)
    # subsample to num agents (so len(.) gives accurate amt)
    AGENT_NAMES_ALL = SAREnv.AGENT_NAMES
    AGENT_NAMES = AGENT_NAMES_ALL[:NUM_AGENTS]

    # useful variables
    FIRE_TYPES = SAREnv.FIRE_TYPES
    EXTINGUISH_TYPES = SAREnv.EXTINGUISH_TYPES
    AMT_FIRE_TYPES = len(FIRE_TYPES)
    INVENTORY_CAPACITY = SAREnv.INVENTORY_CAPACITY
    INVENTORY_TYPES = SAREnv.INVENTORY_TYPES
    MIN_REQUIRED_AGENTS = SAREnv.MIN_REQUIRED_AGENTS
    ALL_INTENSITIES = SAREnv.ALL_INTENSITIES
    CRITICAL_INTENSITY = SAREnv.CRITICAL_INTENSITY
    L_TO_M = SAREnv.L_TO_M
    M_TO_H = SAREnv.M_TO_H
    CARDINAL_DIRECTIONS = SAREnv.CARDINAL_DIRECTIONS
    GRID_WIDTH = SAREnv.GRID_WIDTH
    GRID_HEIGHT = SAREnv.GRID_HEIGHT  

    ENV_STR = f"""The environment consists of fires and lost persons, along with reservoirs, deposits, and robots (you). All in a grid with width {GRID_WIDTH} and height {GRID_HEIGHT}.

    Initially, the robots can see all the fires, but does not know the location of any of the lost people - robots must explore.
    The fires can be of {AMT_FIRE_TYPES} different types: {join_conjunction(FIRE_TYPES, 'or')}, each requiring a different resource to extinguish - {join_conjunction(EXTINGUISH_TYPES, 'and')} respectively. Make sure you use the proper resource to do so.
    A fire consists of a group of 'flammable' objects with intensities of {join_conjunction(ALL_INTENSITIES, 'or')}. It is divided into different regions geographically, so all regions that aren't extinguished (intensity {ALL_INTENSITIES[0]}), must be properly addressed before fire can be extinguished. The first few regions (1,2,etc) are the sources of the fire, and must be addressed first.
    At each step, if a flammable object has an intensity of {ALL_INTENSITIES[1]} or higher, it'll increase in intensity if not extinguished. They spread quickly, so it's important for almost all robots to work collectively to stop the fire.
    In {L_TO_M} steps, the flammable object will go from {ALL_INTENSITIES[1]} to {ALL_INTENSITIES[2]}. In {M_TO_H} steps, the flammable object will go from {ALL_INTENSITIES[2]} to {ALL_INTENSITIES[3]}.
    Once a flammable object reaches an intensity of {CRITICAL_INTENSITY} and not before, it spreads to its immediate neighboors (neighboors with intensity {ALL_INTENSITIES[0]} start with intensity of {ALL_INTENSITIES[1]}).  In order to extinguish a fire, a robot can use the appropriate extinguish resource at that location.
    Then, all the flammable objects in or immediately around this location will lower in intensity by one notch (e.g. from {ALL_INTENSITIES[2]} to {ALL_INTENSITIES[1]}, or {ALL_INTENSITIES[3]} to {ALL_INTENSITIES[2]}).

    The reservoirs can be of type {join_conjunction(EXTINGUISH_TYPES, 'or')}, resources can only be collected at a rate of 1-unit per step.
    Thus, to get more resources, you have to collect the resources multiple times.

    The deposits can hold any amount and type of resources: {join_conjunction(INVENTORY_TYPES, 'and')}.
    The robots can store their entire inventory into the deposit in order to save it for other robots to use.
    When a robot gets a certain resource type (if any is available) from a deposit, the space left in their inventory is filled with that resource type.
    Deposits create an unnecessary middle-step when using them to store resources, so do not waste time in this way.

    If any robot enters the area of visibility of a lost person, that person is found and all robots can now see it.
    Once a person is found, at least {MIN_REQUIRED_AGENTS} robots are required to carry it (could be more depending on person). Otherwise, the person cannot be moved.
    A carried person should be dropped into a deposit (any suffices).
    To drop a carried person, all agents should have navigated to deposit, and they must ALL perform the DropOff action.

    The robots have an inventory capacity of {INVENTORY_CAPACITY} with slots for {join_conjunction(INVENTORY_TYPES, 'and')}."""

    OBS_STR = f"""You will get a description of the task robots are supposed to do. You will get an textual description of the environment from the perspective of {join_conjunction(AGENT_NAMES, 'and')} as the observation input. You will also get a list of objects each robot is able to see in the environment. Here the objects will have a distinct name which will also include which type of object it is.
    So, along with the observation inputs you will get the following information:
    """

    PLANNER_OBS_STR = ",\n".join(
        [
            f"{name}'s observation: local observation (from up, down, left, right, center), global observation, and a list of objects {name} is observing"
            for name in AGENT_NAMES
        ]
    )

    PLANNER_PROMPT = f"""
    You are an excellent planner who is tasked with helping {len(AGENT_NAMES)} embodied robots named {join_conjunction(AGENT_NAMES, 'and')} to carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

    {ENV_STR}

    {OBS_STR}
    ### INPUT FORMAT ###
    {{Task: description of the task the robots are supposed to do,
    {PLANNER_OBS_STR},
    Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
    Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
    Robots' combined memory: description of robots' combined memory}}

    Reason over the robots' task, image inputs, observations, open subtasks, completed subtasks and memory, and then output the following:
    * Reason: The reason for why new subtasks need to be added.
    * Subtasks: A list of open subtasks the robots are supposed to take to complete the task. Remember, as you get new information about the environment, you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed.
    The "Plan" should be in a list format where the actions are listed sequentially.
    For example:
         ["extinguish ChicagoFire_Region_1 using water", "extinguish ChicagoFire_Region_2 using water", "extinguish all of ChicagoFire using water", "collect sufficient water form reservoir"]
         ["locate the lost person", "carry the lost person", "navigate to deposit with lost person", "drop lost person in deposit"]

    Your output should be in the form of a python dictionary as shown below.
    Example output: {{
    "reason": "since the subtask list is empty, the robots need to extinguish the fire, and find & drop the lost person. Thus, for the fire we have to get water from the reservoir and extinguish the Chicago fire, using the deposit if needed to store resources. For the person, we have to locate them by exploring, carry them with enough agents, go to a deposit, and drop the person in it.",
    "plan": ["extinguish ChicagoFire_Region_1 using water", "extinguish ChicagoFire_Region_2 using water", "extinguish all of ChicagoFire using water", "locate the lost person", "carry the lost person", "navigate to deposit with lost person", "drop lost person in deposit"]
    }}

    Ensure that the subtasks are not generic statements like "explore the environment" or "do the task". They should be specific to the task at hand.
    Do not assign subtasks to any particular robot. Try not to modify the subtasks that already exist in the open subtasks list. Rather add new subtasks to the list.

    * NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
    Let's work this out in a step by step way to be sure we have the right answer.
    """

    FAILURE_REASON = """
    If any robot's previous action failed, use the previous history, your current knowledge of the room (i.e. what things are where), and your understanding of causality to think and rationalize about why the previous action failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
    Common failure reasons to lookout for include:
    Trying to collect a resource before navigating to reservoir first,
    trying to use a resource before navigating to fire location first,
    not being close enough to interact with object.
    """

    action_wrapper = lambda name, action: f'"{name}\'s action" : "{action}"'

    # previous actions (that failed) -> ["DropOff(LostPersonJeremy, TheDeposit)", "Idle", "UseSupply(ChicagoFire, Sand)"]
    action_agents_1 = ["NavigateTo(TheDeposit)", "Idle", "UseSupply(ChicagoFire, Water)"]
    ACTION_1 = ",\n".join(
        action_wrapper(AGENT_NAMES_ALL[i], action_agents_1[i]) for i in range(3)
    )

    # ----- example 1 (failure) ------
    # "failure reason" - 1
    FAILURE_REASON_EX_1 = "".join(
        [
            f"{AGENT_NAMES_ALL[0]} and {AGENT_NAMES_ALL[1]} failed to drop off LostPersonJeremy in TheDeposit because {AGENT_NAMES_ALL[1]} had not navigated to the deposit yet, and thus wasn't close enough to interact with it; they both have to be close enough to deposit.",
            f"{AGENT_NAMES_ALL[2]} failed to use the sand supply on the ChicagoFire because the fire is non-chemical, so it requires water.",
        ]
    )

    # "memory" - 1
    MEMORY_EX_1 = " ".join(
        [
            f"{AGENT_NAMES_ALL[0]} finished trying to DropOff LostPersonJeremy at the TheDeposit when {AGENT_NAMES_ALL[0]} was at co-ordinates (4,4).",
            f"{AGENT_NAMES_ALL[1]} finished being idle when {AGENT_NAMES_ALL[1]} was at co-ordinates (14, 6).",
            f"{AGENT_NAMES_ALL[2]} finished using sand supply on the ChicagoFire when {AGENT_NAMES_ALL[2]} was at co-ordinates (7, 24).",
        ]
    )

    # "reason" - 1
    REASON_EX_1 = " ".join(
        [
            f"{AGENT_NAMES_ALL[0]} can wait for {AGENT_NAMES_ALL[1]} to finish navigating to TheDeposit.",
            f"{AGENT_NAMES_ALL[1]} can navigate to TheDeposit in order to be close enough to DropOff LostPersonJeremy.",
            f"{AGENT_NAMES_ALL[2]} can go to use their water supply instead on the ChicagoFire.",
        ]
    )

    # "subtask" - 1
    SUBTASK_EX_1 = " ".join(
        [
            f"{AGENT_NAMES_ALL[0]} is currently waiting for {AGENT_NAMES_ALL[1]} to finish navigating,",
            f"{AGENT_NAMES_ALL[1]} is currently navigating to TheDeposit,",
            f"{AGENT_NAMES_ALL[2]} is currently navigating to the EasternFire,",
        ]
    )


    # -- construct failure example from this ---
    FAILURE_EXAMPLE = f"""
    Example:
    {{
    "failure reason": "{FAILURE_REASON_EX_1}",
    "memory": "{MEMORY_EX_1}",
    "reason": "{REASON_EX_1}",
    "subtask": "{SUBTASK_EX_1}",
    {ACTION_1}
    }}
    """

    ACTION_OBS_STR = ", ".join(
        [
            f"{name}'s observation: list of objects the {name} is observing,\n{name}'s state: description of {name}'s state,\n{name}'s previous action: description of what {name} did in the previous time step and whether it was successful,\n{name}'s previous failures: if {name}'s few previous actions failed, description of what failed,"
            for name in AGENT_NAMES
        ]
    )

    # details for actor
    DETAILS_STR = f"""
    Important details described below:
        * Even if the robot can see an object, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigates to the objects they want to interact with.
        * When navigating to fire, please specify which specific region of the fire you wish to target.
        * Additionally, when you use the supply, it will be dropped wherever you are, NOT in the region you said. So, make sure to navigate to wherever you wish to use a supply.
        * When a fire has an average intensity of {ALL_INTENSITIES[0]}, it means all the flammable objects have been extinguished completely.
        * When the person is not initially visible, you must Explore.
        * When a person is carried, no other action other than navigation and "DropOff" can be made by any of the robots carrying it.
        * When a robot carries a person, all their other resources are dropped and the person takes the entire inventory space.
        * When a group robot wants to drop a person, they must all navigate to the deposit strictly before dropping them. Then, they should ALL perform the DropOff action.
        * When a robot is successful in carrying a person, that just means that specific robot is carrying it, but the person might still not be moveable if an insufficient amount of robots is carrying it.
        * A fire is divided into different regions, and the fire itself (which is just the center). However, there might be flammable objects from the fire that aren't immediately neighbooring this location, so the robot might have to move in different directions to reach them.
        * The resources in the inventory can only be used on fires one unit at a time, so do multiple UseSupply until you clear our your inventory.
        * When a robot is doing a "Move(<direction>)" action, if there is an obstacle, they will not be able to move in that direction.
    """

    ACTION_PROMPT = f"""
    You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {join_conjunction(AGENT_NAMES, 'and')} carry out a task. All {len(AGENT_NAMES)} robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

    {ENV_STR}

    They can perform the following actions:
    ["navigate to object <object_name>", "move <direction>", "explore", "carry <person_name>", "dropoff <person_name> at <deposit_name>", "store supply <deposit_name>", "use supply <resource_name> on <fire_name>", "get supply <resource_name> from <deposit_name>", "get supply <reservoir_name>", "clear inventory", "Done"]

    Here <direction> is one of {str(CARDINAL_DIRECTIONS)}.
    Here <resource_name> is one of {str(EXTINGUISH_TYPES)}.
    The other names (<object_name>, <person_name>, <deposit_name>, <reservoir_name>) are based on the observations.
    When finished with all the subtasks, output "Done" for all agents.

    You need to suggest the action that each robot should take at the current time step.

    {OBS_STR}
    ### INPUT FORMAT ###
    {{Task: description of the task the robots are supposed to do,
    {ACTION_OBS_STR}
    Robots' open subtasks: list of subtasks  supposed to carry out to finish the task. If no plan has been already created, this will be None.
    Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
    Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
    Robots' combined memory: description of robot's combined memory}}

    First of all you are supposed to reason over the image inputs, the robots' observations, previous actions, previous failures, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things:
    * Failure reason: {FAILURE_REASON}
    * Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.
    * Reason: The reasoning for what each robot is supposed to do next
    * Subtask: The subtask each robot should currently try to solve, choose this from the list of open subtasks.
    * Actions for join_conjunction(AGENT_NAMES, 'and'): The actions the robots are supposed to take just in the next step such that they make progress towards completing the task. Make sure that this suggested actions make these robots more efficient in completing the task as compared only one agent solving the task, avoid having agents being idle. Notably, make sure that different agents address different problems (e.g. different fires)
    Your output should just be in the form of a python dictionary as shown below.

    Example of output for 3 robots (do these for {len(AGENT_NAMES)} robots):
    {FAILURE_EXAMPLE}
    Note that the output should just be a dictionary similar to the example output.

    {DETAILS_STR}
    """

    CORRECTOR_PROMPT = f"""
    You are an expert multi-robot team diagnostic and action corrector. Your job is to analyze the most recent step in the Search and Rescue (SAR) environment, identify why any agent actions failed, and recommend the exact next actions that will address those failures, moving the team closer to completing their subtasks efficiently.

    You will receive as input:
    {{
        "Task": <description of the current task>,
        "Last actions": <list of each agent's most recent actions>,
        "Why they failed": <diagnosis of failures for each agent, or "None" if no failure>,
        "Memory": <summary of relevant prior context>,
        "Open subtasks": <list of subtasks that still need to be completed>
    }}

    **Your job:**
    - Analyze each agent’s failure reason in the context of the task and memory.
    - For each agent that failed, recommend a corrected, concrete action (e.g. "navigate to ReservoirUtah", "collect sand", "move north").
    - Ensure all corrected actions are feasible and directly address the reason for failure.
    - If an agent’s last action did not fail, assign the next most logical step toward completing the open subtasks.
    - Briefly update the shared memory if needed (e.g., note new discoveries, outcomes, or changes relevant for future steps).

    **Your output MUST be a JSON object in a fenced code block:**
    ```json
    {{
        "corrected_actions": ["<agent_1 action>", "<agent_2 action>", ...],
        "memory": "<updated shared memory for the team>"
    }}

    Do not output anything except this JSON block. Think step by step to ensure all corrections are accurate and efficient.
    """

    VERIFIER_OBS_STR = ",\n".join(
        [
            f"{name}'s observation: list of objects the {name} is observing,\n{name}'s state: description of {name}'s state,\n{name}'s previous action: the action {name} took in the previous step,"
            for name in AGENT_NAMES
        ]
    )

    VERIFIER_PROMPT = f"""You are an excellent planner who is tasked with helping {len(AGENT_NAMES)} embodied robots named {join_conjunction(AGENT_NAMES, 'and')} to carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

    {ENV_STR}

    {OBS_STR}
    ### INPUT FORMAT ###
    {{Task: description of the task the robots are supposed to do,
    {VERIFIER_OBS_STR}
    Robots' open subtasks: list of open subtasks the robots in the previous step. If no plan has been already created, this will be None,
    Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None,
    Robots' combined memory: description of robots' combined memory
    }}

    You will receive the following information:
    * Reason: The reason for why you think a particular subtask should be moved from the open subtasks list to the completed subtasks list.
    * Completed Subtasks: The list of subtasks that have been completed by the robots. Note that you can add subtasks to this list only if they have been successfully completed and were in the open subtask list. If no subtasks have been completed at the current step, return an empty list.

    The "Completed Subtasks" should be in a list format where the completed subtasks are listed. For example: ["collect water from deposit", "collect sufficient sand from reservoir"]
    Your output should be in the form of a python dictionary as shown below.

    Example output with two agents (do it for {len(AGENT_NAMES)} agents):
    {{"reason": "{AGENT_NAMES[0]} used water on the ChicagoFire in the previous step and was successful, and {AGENT_NAMES[1]} explored and was successful. Since the ChicagoFire is still not extinguished completely, {AGENT_NAMES[0]} has still not completed the subtask of extinguishing ChicagoFire using water. Since the lost person is now visible {AGENT_NAMES[1]} has completed the subtask of finding the lost person.",
    "completed subtasks": ["locate lost person"]
    }}

    When you output the completed subtasks, make sure to not forget to include the previous ones in addition to the new ones.
    Also, make sure to never add a subtask to completed subtasks before it has successfully been completed.
    Let's work this out in a step by step way to be sure we have the right answer.

    * NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
    """
    return (
        ACTION_PROMPT,
        CORRECTOR_PROMPT,
        PLANNER_PROMPT,
        VERIFIER_PROMPT,
        extract_json_block,
        process_action_llm_output,
    )


@app.cell
def _(LLAMARState, PLANNER_PROMPT, baseenv, extract_json_block, json, openai):
    def planner_node(state: LLAMARState) -> LLAMARState:
        user_dict = {
            "Task": state["task_description"],
            **{f"{ag}'s observation": state["agent_states"][ag]["observation"]
               for ag in state["agent_states"]},
            "Robots' open subtasks": state["open_subtasks"],
            "Robots' completed subtasks": state["completed_subtasks"],
            "Robots' combined memory": state["memory"],
        }
        user_prompt = baseenv.convert_dict_to_string(user_dict)
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":PLANNER_PROMPT},
                      {"role":"user","content":user_prompt}]
        )
        plan = json.loads(extract_json_block(resp.choices[0].message.content))["plan"]
        # Update shared state
        state["plan"] = plan
        state["open_subtasks"] = plan
        return state
    return (planner_node,)


@app.cell
def _(
    ACTION_PROMPT,
    LLAMARState,
    baseenv,
    extract_json_block,
    json,
    openai,
    process_action_llm_output,
):
    def actor_node(state: LLAMARState) -> LLAMARState:
        user_dict = { "Task": state["task_description"] }
        for ag, slice in state["agent_states"].items():
            user_dict.update({
                f"{ag}'s observation": slice["observation"],
                f"{ag}'s state": slice["state"],
                f"{ag}'s previous action": slice["previous_action"],
                f"{ag}'s previous failures": slice["previous_failures"],
            })
        user_dict.update({
            "Robots' open subtasks": state["open_subtasks"],
            "Robots' completed subtasks": state["completed_subtasks"],
            "Robots' combined memory": state["memory"],
        })
        if state["corrective_action"]:
            user_dict["Corrective action"] = state["corrective_action"]
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":ACTION_PROMPT},
                      {"role":"user","content":baseenv.convert_dict_to_string(user_dict)}]
        )
        out = json.loads(extract_json_block(resp.choices[0].message.content))
        actions, reason, subtask, memory, failure_reason = process_action_llm_output(out)
        # Update shared state
        state["last_actions"] = actions
        state["memory"] = memory
        # stash current subtask focus for verifier
        state["agent_states"][list(state["agent_states"].keys())[0]]["subtask"] = subtask
        state["agent_states"][list(state["agent_states"].keys())[0]]["failure_reason"] = failure_reason
        return state
    return (actor_node,)


@app.cell
def _(CORRECTOR_PROMPT, LLAMARState, extract_json_block, json, openai):
    def corrector_node(state: LLAMARState) -> LLAMARState:
        if not state["agent_states"][list(state["agent_states"])[0]].get("failure_reason"):
            # nothing failed – skip
            return state
        user_ctx = {
            "Task": state["task_description"],
            "Last actions": state["last_actions"],
            "Why they failed": state["agent_states"][list(state["agent_states"])[0]]["failure_reason"],
            "Memory": state["memory"],
            "Open subtasks": state["open_subtasks"],
        }
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":CORRECTOR_PROMPT},
                      {"role":"user","content":json.dumps(user_ctx, indent=2)}]
        )
        correction_dict = json.loads(extract_json_block(resp.choices[0].message.content))
        state["corrective_action"] = correction_dict["corrected actions"]
        state["memory"] = correction_dict["memory"]
        return state
    return (corrector_node,)


@app.cell
def _(LLAMARState, VERIFIER_PROMPT, baseenv, extract_json_block, json, openai):
    def verifier_node(state: LLAMARState) -> LLAMARState:
        user_dict = {
            "Task": state["task_description"],
            **{f"{ag}'s observation": slice["observation"]
               for ag, slice in state["agent_states"].items()},
            **{f"{ag}'s previous action": slice["previous_action"]
               for ag, slice in state["agent_states"].items()},
            "Robots' open subtasks": state["open_subtasks"],
            "Robots' completed subtasks": state["completed_subtasks"],
            "Robots' combined memory": state["memory"],
        }
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":VERIFIER_PROMPT},
                      {"role":"user","content":baseenv.convert_dict_to_string(user_dict)}]
        )
        result = json.loads(extract_json_block(resp.choices[0].message.content))
        state["completed_subtasks"] = list(set(state["completed_subtasks"]) |
                                           set(result["completed subtasks"]))
        # prune from open list
        state["open_subtasks"] = [st for st in state["open_subtasks"]
                                  if st not in state["completed_subtasks"]]
        return state
    return (verifier_node,)


@app.cell
def _(
    END,
    LLAMARState,
    StateGraph,
    actor_node,
    corrector_node,
    planner_node,
    verifier_node,
):
    g=StateGraph(LLAMARState)

    g.add_node("Planner", planner_node)
    g.add_node("Actor", actor_node)
    g.add_node("Verifier", verifier_node)
    g.add_node("Corrector", corrector_node)

    # unconditional edges
    g.add_edge("Planner","Actor")
    g.add_edge("Actor","Verifier")
    # conditional: Actor → Corrector when failure_reason present
    g.add_conditional_edges(
        "Actor",
        lambda s: "Corrector" if any(agent.get("failure_reason") for agent in s["agent_states"].values()) else "Verifier",
        {
            "Corrector": "Corrector",
            "Verifier": "Verifier",
        }
    )

    g.add_edge("Corrector","Actor")  # retry loop
    # Verifier decides to END or re-plan
    g.add_conditional_edges(
        "Verifier",
        lambda s: "END" if not s["open_subtasks"] else "Planner",
        {
            "END": END,
            "Planner": "Planner",
        }
    )

    g.set_entry_point("Planner")
    return (g,)


@app.cell(column=1)
def _(g, mo):
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
    image_data = graph.get_graph().draw_mermaid()
    mo.mermaid(diagram=image_data)
    return


@app.cell
def _():
    # # Cell 6: Graph Invocation and Testing

    # print("--- Invoking Graph ---")
    # initial_state_example = {"messages": [("user", "Hello LangGraph!")], "loop_count": 0, "next_node": ""}
    # for s_idx, s_val in enumerate(graph.stream(initial_state_example)):
    #     print(f"Step {s_idx}: {s_val}")
    # print("--- Graph Invocation Complete ---")
    return


if __name__ == "__main__":
    app.run()
