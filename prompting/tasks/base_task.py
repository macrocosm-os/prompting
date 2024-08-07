import time
import bittensor as bt
from abc import ABC
from pydantic import BaseModel
from prompting.llms.base_llm import BasePipeline
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.utils.cleaners import CleanerPipeline
from typing import ClassVar
from prompting.datasets.base import Context
from abc import abstractmethod


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


class BaseTask(ABC, BaseModel):
    context: dict
    augment: bool = False

    query: str | None = None

    query_system_prompt: ClassVar[str] = ""
    reference_system_prompt: ClassVar[str] = ""
    augmentation_system_prompt: ClassVar[str] = ""

    cleaner: ClassVar[CleanerPipeline] = CleanerPipeline()

    @abstractmethod
    def generate_query_reference(llm_pipeline: BasePipeline, context: Context, **kwargs) -> [str, str]:
        raise NotImplementedError("Method generate_query_reference must be implemented")

    @classmethod
    def generate_reference(cls, llm_pipeline: BasePipeline, messages: list[str]) -> str:
        """Generates a reference answer to be used for scoring miner completions"""
        if len(cls.reference_system_prompt) == 0:
            bt.logging.error("Reference prompt is empty. Please provide a reference prompt.")

        bt.logging.info("🤖 Generating reference...")
        reference = vLLM_LLM(llm_pipeline, system_prompt=cls.reference_system_prompt).query(
            cleaner=cls.cleaner, message=messages
        )
        return reference

    @classmethod
    def generate_query(
        cls,
        messages: str,
        llm_pipeline: BasePipeline,
    ) -> str:
        """Generates a query to be used for generating the challenge"""
        bt.logging.info("🤖 Generating query...")
        query = vLLM_LLM(llm_pipeline, system_prompt=cls.query_system_prompt).query(message=messages)
        return query

    @classmethod
    def augment_query(
        cls,
        query: str,
        llm_pipeline: BasePipeline,
    ) -> str:
        """Creates the opening question of the conversation which is based on the task query but dressed in the persona of the user."""
        challenge = vLLM_LLM(
            llm_pipeline=llm_pipeline,
            max_new_tokens=256,
            system_prompt=cls.augmentation_system_prompt,
        ).query(message=query)
        return challenge
