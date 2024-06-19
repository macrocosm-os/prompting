import time
from typing import Any
import bittensor as bt
from dataclasses import dataclass
from prompting.cleaners.cleaner import CleanerPipeline
from prompting.llms.base_llm import BasePipeline
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.shared.context import Context
from prompting.tasks import Task
from transformers import Pipeline

from prompting.tasks.task import make_system_prompt

# QUERY_SYSTEM_PROMPT = ""
TASK_NAME = "organic"


@dataclass
class OrganicTask(Task):
    name = TASK_NAME
    desc = "get help on answering a question"
    goal = "to get the answer to the following question"
    # Use challenge as a query.
    challenge_type = "query"

    reward_definition = [
        # dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
        dict(name="relevance", weight=1.0),
    ]
    penalty_definition = [
        dict(name="relevance", weight=1.0),
    ]

    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
        dict(name="remove_post_question_text"),
    ]

    def __init__(self, llm_pipeline: Pipeline, context: Context, create_reference: bool = True):
        self.context = context
        self.query = context.content
        self.reference_prompt = context.content
        self.messages = context.messages
        self.roles = context.roles
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)

    def generate(
            self,
            system: str,
            messages: list[str],
            roles: list[str],
            pipeline: BasePipeline,
            clean=True
        ) -> str:
        """Use the LLM to generate a response to a prompt"""
        cleaner = CleanerPipeline(cleaning_pipeline=self.cleaning_pipeline) if clean else None
        return vLLM_LLM(pipeline, system_prompt=system).query_conversation(
            messages=messages, roles=roles, cleaner=cleaner)

    def generate_reference(self, pipeline: BasePipeline, clean=True) -> str:
        """Generates a reference answer to be used for scoring miner completions"""
        t0 = time.perf_counter()
        if not self.static_reference:
            if not self.clean_reference:
                clean = False
            bt.logging.info("🤖 Generating reference...")
            self.reference = self.generate(
                system=make_system_prompt(),
                messages=self.messages,
                roles=self.roles,
                pipeline=pipeline,
                clean=clean,
            )

        self.reference_time = time.perf_counter() - t0
        return self.reference
