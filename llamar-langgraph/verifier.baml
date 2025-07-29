class VerifierAgentInfo {
  observation string
  previous_action string
}

class VerifierInput {
  task_description string
  agent_info map<string, VerifierAgentInfo> // Key: agent_name, Value: VerifierAgentInfo
  open_subtasks string[]
  completed_subtasks string[]
  memory string
  agent_names string[]
  env_consts EnvConstants
}

class VerifierOutput {
  reason string @description("The reason for why a particular subtask should be moved from the open to the completed list.")
  completed_subtasks string[] @description("The list of subtasks that have now been completed.")
}

function VerifySubtasks(input: VerifierInput) -> VerifierOutput {
  client openai/gpt-4o

  prompt #"
    You are an excellent planner tasked with verifying the progress of {{ input.agent_names | length }} robots.

    {{ PrintEnvDetails(input.env_consts) }}

    {{ PrintObserverInfo(input.agent_names) }}

    ### INPUT FORMAT ###
    Task: {{ input.task_description }}
    {% for name, info in input.agent_info.items() %}
    {{ name }}'s observation: {{ info.observation }}
    {{ name }}'s previous action: {{ info.previous_action }}
    {% endfor %}
    Robots' open subtasks: {{ input.open_subtasks | join(', ') }}
    Robots' completed subtasks: {{ input.completed_subtasks | join(', ') }}
    Robots' combined memory: {{ input.memory }}

    Based on the inputs, determine which subtasks from the open list have now been successfully completed.

    Provide a reason and an updated list of all completed subtasks (including previous ones). If no new subtasks were completed, return an empty list for the new completions.

    {{ ctx.output_format }}
  "#
}