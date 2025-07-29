class AgentStateInfo {
  observation string
  state string
  previous_action string
  previous_failures string
}

class ActorInput {
  task_description string
  agent_states map<string, AgentStateInfo> // Key: agent_name, Value: AgentStateInfo
  open_subtasks string[]
  completed_subtasks string[]
  memory string
  agent_names string[]
  env_consts EnvConstants
}

class ActorOutput {
  failure_reason string? @description("Reason why the previous action failed. Null if successful.")
  memory string @description("Important information to remember for the future.")
  reason string @description("The reasoning for what each robot is supposed to do next.")
  subtask string @description("The subtask each robot should currently try to solve.")
  actions map<string, string> @description("A map where the key is the agent's name and the value is the action they should take.")
}

function DecideNextActions(input: ActorInput) -> ActorOutput {
  client openai/gpt-4o

  prompt #"
    You are an excellent planner and robot controller for {{ input.agent_names | length }} robots named {{ input.agent_names | join(' and ') }}.

    {{ PrintEnvDetails(input.env_consts) }}

    They can perform actions like: "navigate to object <object_name>", "move <direction>", "explore", "carry <person_name>", etc. When finished, output "Done" for all agents.

    {{ PrintObserverInfo(input.agent_names) }}

    ### INPUT FORMAT ###
    Task: {{ input.task_description }}
    {% for name, state in input.agent_states.items() %}
    {{ name }}'s observation: {{ state.observation }}
    {{ name }}'s state: {{ state.state }}
    {{ name }}'s previous action: {{ state.previous_action }}
    {{ name }}'s previous failures: {{ state.previous_failures }}
    {% endfor %}
    Robots' open subtasks: {{ input.open_subtasks | join(', ') }}
    Robots' completed subtasks: {{ input.completed_subtasks | join(', ') }}
    Robots' combined memory: {{ input.memory }}

    Reason over the inputs and output a JSON object detailing the failure reason (if any), updated memory, reasoning, current subtask, and the next action for each robot.

    {{ PrintActorDetails(input.env_consts.critical_intensity) }}

    {{ ctx.output_format }}
  "#
}