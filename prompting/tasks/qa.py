from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
import bittensor as bt
from prompting.tasks.task import BaseTask, WeightedRewardModel

# from prompting.rewards.reward import BaseRewardModel
from prompting.utils.cleaners import RemoveRoles, RemoveQuotes, PruneEnding, RemovePostQuestionText
from prompting.utils.cleaners import CleanerPipeline
from prompting.shared import Context
from pydantic import model_validator
from prompting.tasks.task import BaseRewardConfig

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. The questions you generate should be based on the context that is provided.
You will maintain a neutral tone in your questions.
You will adhere to a word limit of 50 words for each question.
"""

# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context:

#Context:
{context}
"""

# Used to obtain the query (which is a followup question about the context)
# TODO: we may not need the entire conversation history - we can sample a subset of it (first k messages, last k messages, etc.)
FOLLOWUP_PROMPT_TEMPLATE = """
Compose a single, specific question to continue the dialogue below. Adopt the persona of the original user, reflecting their communication style and objectives. The question should be rooted in the previous exchanges and should not be answerable with a simple yes or no.

Ensure the question requires detailed knowledge of the conversation history for a correct response, focusing on requests for clarification or additional details (e.g., 'What specific steps did you take?', 'Are you sure?', 'How do you know that is true', or 'How did that situation resolve?'). Use indirect pronouns or descriptions to refer to subjects instead of their names. Avoid answering the question yourself and do not introduce new information not already discussed.

When asking a followup question, you should use pronouns or descriptions to refer to subjects instead of their names. You absolutely must not repeat the subject of the question in the followup question. For example, if the question is "What is the capital of France?", the followup question should not be "What is the population of France?". Instead, it should be "How many people live there?" or "What is its population?".
# Context:
{context}

# Conversation History:
{history}
"""


# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Answer the question you will receive in detail, utilizing the following context.

#Context:
{context}

# Question:
{question}
"""

# TODO: We also need a special followup reference prompt (or just merge both)
# TODO: We should create followups using the specified llama3 chat template rather than feeding the message history through textually
FOLLOWUP_REFERENCE_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question below in detail, prioritizing the use of the provided conversation history. The context is available for additional information if needed, but it may not always be relevant.

# Conversation History:
{history}

# Context (optional):
{context}

# Question:
{question}

Ensure your answer references relevant parts of the conversation history. Use the context only if it provides additional necessary information.
"""


class QARewardConfig(BaseRewardConfig):
    reward_definitions: list[WeightedRewardModel] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]


class QuestionAnsweringTask(BaseTask):
    """QuestionAnsweringTasks must be initialised with an LLM pipeline to generate query and reference plus
    context from a dataset to base the query on"""

    context: Context
    query: str | None = None
    reference: str | None = None
    cleaning_pipeline: CleanerPipeline = CleanerPipeline(
        cleaning_pipeline=[
            RemoveQuotes(),
            PruneEnding(),
            RemoveRoles(),
            RemovePostQuestionText(),
        ]
    )
    create_reference: bool = True
    history: str | None = None

    @model_validator(mode="after")
    def make_query_reference_prompts(self) -> "QuestionAnsweringTask":
        if self.query and self.reference:
            return self

        if self.history:
            self.query_prompt = FOLLOWUP_PROMPT_TEMPLATE.format(context=self.context.content, history=self.history)
            self.reference_prompt = FOLLOWUP_REFERENCE_PROMPT_TEMPLATE.format(
                context=self.context.content, question=self.query, history=self.history
            )
            bt.logging.warning(f"Using history!!\n{self.history=}\n\n{self.context=}\n\n{self.query_prompt=}")
        else:
            self.query_prompt = QUERY_PROMPT_TEMPLATE.format(context=self.context.content)
            self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=self.context.content, question=self.query)

        # self.query = self.generate_query(llm_pipeline, query_prompt=query_prompt)
        # if self.create_reference:
        #     self.reference = self.generate_reference(llm_pipeline, reference_prompt=reference_prompt)
        # self.topic = self.context.title
        # self.subtopic = self.context.topic
        # self.tags = self.context.tags
        return self
