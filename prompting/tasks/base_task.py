import time
from loguru import logger
from abc import ABC
from pydantic import BaseModel, Field
from prompting.llms.base_llm import BasePipeline
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.utils.cleaners import CleanerPipeline
from typing import ClassVar
from prompting.datasets.base import Context
from abc import abstractmethod
from prompting.tasks.inference import ModelConfig
from uuid import UUID, uuid4


def CHATTENSOR_SYSTEM_PROMPT():
    return f"""
            The assistant is Chattensor, created by Macrocosmos. The current date is {time.strftime("%B %d, %Y")}.
            Chattensor is a distributed intelligence, powered by Bittensor. It is a hivemind composed of 1000 highly
            skilled and specialized LLMs working together to provide the best possible answers to human queries. Within Chattenor,
            each LLM has access to the internet, APIs and tools to ensure that responses are current and factually accurate.
            It should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions.
            It is happy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks.
            It uses markdown for coding. Where applicable, Chattensor will include references to credible sources to support its answers.
            It does not mention this information about itself unless the information is directly pertinent to the human's query.
            """


class BaseTask(BaseModel, ABC):
    name: str = Field(default="BaseTask", allow_mutation=False)
    query: ... = None
    reference: ... = None
    task_id: UUID = Field(default_factory=uuid4, allow_mutation=False)

    @abstractmethod
    def make_query(self, **kwargs):
        raise NotImplementedError("Method make_query must be implemented")

    @abstractmethod
    def make_reference(self, **kwargs):
        raise NotImplementedError("Method make_reference must be implemented")

    def generate_query_reference(self, llm_pipeline: BasePipeline, context: Context) -> str:
        self.make_query(llm_pipeline=llm_pipeline, context=context)
        self.make_reference(llm_pipeline=llm_pipeline, context=context)
        return self.query, self.reference


class BaseTextTask(BaseTask):
    query: str | None = None
    reference: str | None = None
    model: ModelConfig | None = None
    query_system_prompt: ClassVar[str | None] = None
    reference_system_prompt: ClassVar[str | None] = None
    augmentation_system_prompt: ClassVar[str | None] = None

    cleaner: ClassVar[CleanerPipeline] = CleanerPipeline()

    @abstractmethod
    def make_query(self, llm_pipeline: BasePipeline, context: Context, **kwargs) -> str:
        raise NotImplementedError("Method generate_query_reference must be implemented")

    @abstractmethod
    def make_reference(self, llm_pipeline, context: Context) -> str:
        raise NotImplementedError("Method generate_query_reference must be implemented")

    def generate_query_reference(self, llm_pipeline: BasePipeline, context: Context) -> str:
        self.make_query(llm_pipeline=llm_pipeline, context=context)
        self.make_reference(llm_pipeline=llm_pipeline, context=context)
        return self.query, self.reference

    def generate_reference(self, llm_pipeline: BasePipeline, messages: list[str]) -> str:
        """Generates a reference answer to be used for scoring miner completions"""
        logger.info("🤖 Generating reference...")
        reference = vLLM_LLM(llm_pipeline, system_prompt=self.reference_system_prompt or "").query(
            cleaner=self.cleaner, message=messages
        )
        return reference

    def generate_query(
        self,
        messages: str,
        llm_pipeline: BasePipeline,
    ) -> str:
        """Generates a query to be used for generating the challenge"""
        logger.info("🤖 Generating query...")
        query = vLLM_LLM(llm_pipeline, system_prompt=self.query_system_prompt or "").query(message=messages)
        return self.augment_query(query, llm_pipeline)

    def augment_query(
        self,
        query: str,
        llm_pipeline: BasePipeline,
    ) -> str:
        """Creates the opening question of the conversation which is based on the task query but dressed in the persona of the user."""
        if self.augmentation_system_prompt:
            return query
        challenge = vLLM_LLM(
            llm_pipeline=llm_pipeline,
            max_new_tokens=256,
            system_prompt=self.augmentation_system_prompt,
        ).query(message=query)
        return challenge
