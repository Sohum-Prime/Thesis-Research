// We use maps (dictionaries) to handle a dynamic number of agents without changing the schema.

class PlannerInput {
  task_description string
  agent_observations map<string, string> // Key: agent_name, Value: observation string
  open_subtasks string[]
  completed_subtasks string[]
  memory string
  agent_names string[]
  env_consts EnvConstants
}

class PlannerOutput {
  reason string @description("The reason for why new subtasks need to be added.")
  plan string[] @description("A list of open subtasks the robots are supposed to take to complete the task.")
}

function GeneratePlan(input: PlannerInput) -> PlannerOutput {
  client openai/gpt-4o

  prompt #"
    You are an excellent planner who is tasked with helping {{ input.agent_names | length }} embodied robots named {{ input.agent_names | join(' and ') }} to carry out a task.

    {{ PrintEnvDetails(input.env_consts) }}

    {{ PrintObserverInfo(input.agent_names) }}

    ### INPUT FORMAT ###
    Task: {{ input.task_description }}
    {% for name, obs in input.agent_observations.items() %}
    {{ name }}'s observation: {{ obs }}
    {% endfor %}
    Robots' open subtasks: {{ input.open_subtasks | join(', ') }}
    Robots' completed subtasks: {{ input.completed_subtasks | join(', ') }}
    Robots' combined memory: {{ input.memory }}

    Reason over the robots' task, observations, subtasks and memory, and then output a JSON object with your reasoning and a new plan.

    Do not include the subtasks that have already been completed.
    The "plan" should be in a list format where the actions are listed sequentially.
    Ensure that the subtasks are not generic statements like "explore the environment". They should be specific to the task at hand.

    {{ ctx.output_format }}
  "#
}