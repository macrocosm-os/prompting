from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.tasks.base_task import BaseTask
from prompting.rewards.reward import WeightedRewardModel

# from prompting.rewards.reward import BaseRewardModel
from prompting.utils.cleaners import RemoveRoles, RemoveQuotes, PruneEnding, RemovePostQuestionText
from prompting.utils.cleaners import CleanerPipeline
from prompting.datasets.base import Context
from prompting.rewards.reward import BaseRewardConfig
from typing import ClassVar

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. The questions you generate should be based on the context that is provided.
You will maintain a neutral tone in your questions.
You will adhere to a word limit of 50 words for each question.
"""

REFERENCE_SYSTEM_PROMPT = """\
You are an expert question-answering LLM. You will receive context and a question, and you will generate a detailed and accurate answer to the question. Your answer should be based on the context provided.
"""

# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context:

#Context:
{context}
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
    penalty_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
    ]

    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
        dict(name="remove_post_question_text"),
    ]


class QuestionAnsweringTask(BaseTask):
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
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    reference_system_prompt: ClassVar[str] = REFERENCE_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""

    @classmethod
    def generate_query_reference(cls, llm_pipeline, context: Context):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.content)
        query = cls.generate_query(llm_pipeline=llm_pipeline, messages=[query_prompt])
        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=context.content, question=query)
        reference = cls.generate_reference(lllm_pipeline=llm_pipeline, messages=[reference_prompt])
        return query, reference

    # @model_validator(mode="after")
    # def make_query_reference_prompts(self) -> "QuestionAnsweringTask":
    #     if self.query and self.reference:
    #         return self

    #     if self.history:
    #         self.query_prompt = FOLLOWUP_PROMPT_TEMPLATE.format(context=self.context.content, history=self.history)
    #         self.reference_prompt = FOLLOWUP_REFERENCE_PROMPT_TEMPLATE.format(
    #             context=self.context.content, question=self.query, history=self.history
    #         )
    #         bt.logging.warning(f"Using history!!\n{self.history=}\n\n{self.context=}\n\n{self.query_prompt=}")
    #     else:
    #         self.query_prompt = QUERY_PROMPT_TEMPLATE.format(context=self.context.content)
    #         self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=self.context.content, question=self.query)

    #     # self.query = self.generate_query(llm_pipeline, query_prompt=query_prompt)
    #     # if self.create_reference:
    #     #     self.reference = self.generate_reference(llm_pipeline, reference_prompt=reference_prompt)
    #     # self.topic = self.context.title
    #     # self.subtopic = self.context.topic
    #     # self.tags = self.context.tags
    #     return self
