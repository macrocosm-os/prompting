from typing import ClassVar

from prompting.datasets.base import Context
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.rewards.rouge import RougeRewardModel
from prompting.tasks.base_task import BaseTextTask
from prompting.utils.cleaners import CleanerPipeline, PruneEnding, RemovePostQuestionText, RemoveQuotes, RemoveRoles
from prompting.tasks.qa import QuestionAnsweringTask
from prompting.llms.apis.gpt_wrapper import LLMMessage, LLMMessages
from prompting.llms.apis.llm_wrapper import LLMWrapper
from loguru import logger
from prompting.utils.timer import Timer
import json
import time

def make_api_call(messages, max_tokens, is_final_answer=False):
    # TOOD: Make this use local model to prevent relay mining
    response = LLMWrapper.chat_complete(messages=LLMMessages(*messages))
    return json.loads(response)

def generate_response(prompt):
    messages = [LLMMessage(role="system", content="""You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or provide a final answer.

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Key Instructions:
- Employ at least 5 distinct reasoning steps.
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 3 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.


Example of a valid JSON response:
```json
{
    "title": "Initial Problem Analysis",
    "content": "To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.",
    "next_action": "continue"
}```
""")]               
    messages += [LLMMessage(role="user", content=prompt)]
    messages += [LLMMessage(role="assistant", content="Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem.")]


    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        with Timer() as timer:
            step_data = make_api_call(messages, 300)
        thinking_time = timer.final_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data['next_action'] == 'final_answer':
            break

        step_count += 1

        # Yield after each step
        yield steps, None

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})

    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data['content'], thinking_time))

    yield steps, total_thinking_time


def execute_multi_step_reasoning(user_query):
    for steps, total_thinking_time in generate_response(user_query):
        if total_thinking_time is not None:
            logger.info(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
        return steps



class MultiStepReasoningRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        RelevanceRewardModel(weight=1),
    ]


class MultiStepReasoningTask(QuestionAnsweringTask):
    """QuestionAnsweringTasks must be initialised with an LLM pipeline to generate query and reference plus
    context from a dataset to base the query on"""

    cleaning_pipeline: ClassVar[CleanerPipeline] = CleanerPipeline(
        cleaning_pipeline=[
            RemoveQuotes(),
            PruneEnding(),
            RemoveRoles(),
            RemovePostQuestionText(),
        ]
    )
    name: ClassVar[str] = "multi_step_reasoning"
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    reference_system_prompt: ClassVar[str] = REFERENCE_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""
    query: str | None = None
    reference: str | None = None

    def make_reference(self, dataset_entry: Context):
        steps, total_thinking_time = execute_multi_step_reasoning(user_query=self.query)
        logger.info(f"**Total thinking time for multi step reasoning: {total_thinking_time:.2f} seconds**")
        self.reference = steps[-1][1]
        return self.reference
