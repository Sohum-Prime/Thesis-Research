class CorrectorInput {
  task_description string
  last_actions string[]
  failure_diagnosis string @description("Diagnosis of failures for each agent, or 'None' if no failure.")
  memory string
  open_subtasks string[]
}

class CorrectorOutput {
  corrected_actions string[]
  memory string
}

function CorrectFailedActions(input: CorrectorInput) -> CorrectorOutput {
  client openai/gpt-4o

  prompt #"
    You are an expert multi-robot team diagnostic and action corrector. Your job is to analyze the most recent step in the Search and Rescue (SAR) environment, identify why any agent actions failed, and recommend the exact next actions that will address those failures.

    ### INPUT ###
    Task: {{ input.task_description }}
    Last actions: {{ input.last_actions | join(', ') }}
    Why they failed: {{ input.failure_diagnosis }}
    Memory: {{ input.memory }}
    Open subtasks: {{ input.open_subtasks | join(', ') }}

    ### YOUR JOB ###
    - Analyze each agent’s failure reason in the context of the task and memory.
    - For each agent that failed, recommend a corrected, concrete action.
    - If an agent’s last action did not fail, assign the next most logical step toward completing the open subtasks.
    - Briefly update the shared memory if needed.

    Your output MUST be a JSON object.

    {{ ctx.output_format }}
  "#
}