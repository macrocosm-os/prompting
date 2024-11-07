import time
from typing import Any
from loguru import logger
from abc import ABC
from pydantic import BaseModel, Field, ConfigDict, model_validator
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.utils.cleaners import CleanerPipeline
from typing import ClassVar
from prompting.datasets.base import DatasetEntry
from abc import abstractmethod
from uuid import uuid4
from prompting.llms.model_zoo import ModelConfig
from prompting.llms.model_manager import model_manager
import random
from prompting.settings import settings
from prompting.llms.apis.gpt_wrapper import LLMMessage, LLMMessages
from prompting.llms.apis.llm_wrapper import LLMWrapper


def CHATTENSOR_SYSTEM_PROMPT():
    return f"""
            You are Chattensor, an assistant created by Macrocosmos. The current date is {time.strftime("%B %d, %Y")}.
            Chattensor is a distributed intelligence, powered by Bittensor. It is a hivemind composed of 1000 highly
            skilled and specialized LLMs working together to provide the best possible answers to human queries. Within Chattenor,
            each LLM has access to the internet, APIs and tools to ensure that responses are current and factually accurate.
            It should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions.
            It is happy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks.
            It uses markdown for coding. Where applicable, Chattensor will include references to credible sources to support its answers.
            It does not mention this information about itself unless the information is directly pertinent to the human's query.
            """


class BaseTask(BaseModel, ABC):
    name: str | None = None
    query: Any = None
    reference: Any = None
    task_id: str = Field(default_factory=lambda: str(uuid4()), allow_mutation=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def make_query(self, **kwargs):
        raise NotImplementedError("Method make_query must be implemented")

    @abstractmethod
    def make_reference(self, **kwargs):
        raise NotImplementedError("Method make_reference must be implemented")


class BaseTextTask(BaseTask):
    query: str | None = None
    messages: list[str] | None = None
    reference: str | None = None
    llm_model: ModelConfig = None
    llm_model_id: str = None
    seed: str = None
    query_system_prompt: ClassVar[str | None] = None
    reference_system_prompt: ClassVar[str | None] = None
    augmentation_system_prompt: ClassVar[str | None] = None
    dataset_entry: DatasetEntry | None = None
    task_id: str = str(uuid4())

    cleaner: ClassVar[CleanerPipeline] = CleanerPipeline()

    @model_validator(mode="after")
    def get_model_id_and_seed(self) -> "BaseTextTask":
        if self.llm_model:
            self.llm_model_id = self.llm_model.llm_model_id if self.llm_model else None
            self.seed = random.randint(0, 1000000)
        return self

    def make_query(self, dataset_entry: DatasetEntry, **kwargs) -> str:
        return self.query

    def make_reference(self, dataset_entry: DatasetEntry) -> str:
        return self.reference

    def generate_reference(self, messages: list[str]) -> str:
        """Generates a reference answer to be used for scoring miner completions"""
        logger.info("🤖 Generating reference...")
        self.reference = vLLM_LLM(
            llm=model_manager.get_model(self.llm_model).llm, system_prompt=self.reference_system_prompt or ""
        ).query(cleaner=self.cleaner, message=messages)
        # self.reference = model_manager.get_model(self.llm_model).generate(prompts=messages)
        if self.reference is None:
            raise Exception("Reference generation failed")
        return self.reference

    def generate_query(
        self,
        messages: list[str],
    ) -> str:
        """Generates a query to be used for generating the challenge"""
        logger.info("🤖 Generating query...")
        llm_messages = [LLMMessage(role="system", content=self.query_system_prompt)] if self.query_system_prompt else []
        llm_messages += [LLMMessage(role="user", content=message) for message in messages]

        self.query = LLMWrapper.chat_complete(messages=LLMMessages(*llm_messages))

        if self.query is None:
            raise Exception("Query generation failed")
        return self.augment_query(self.query)

    def augment_query(
        self,
        query: str,
    ) -> str:
        """Creates the opening question of the conversation which is based on the task query but dressed in the persona of the user."""
        if not self.augmentation_system_prompt:
            return query
        challenge = LLMWrapper.chat_complete(
            messages=LLMMessages(
                LLMMessage(role="system", content=self.augmentation_system_prompt),
                LLMMessage(role="user", content=query),
            ),
            max_tokens=settings.NEURON_MAX_TOKENS,
        )
        self.query = challenge
        return challenge
