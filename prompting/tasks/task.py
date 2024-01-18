import time
import bittensor as bt
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import List, Union
from prompting.llm import HuggingFaceLLM
from transformers import Pipeline

from prompting.utils.clean_generation import GenerationCleaner


class TaskEvaluationType(Enum):
    REWARD_STACK = "reward"
    FILTER_STACK = "filter"
    PENALTY_STACK = "penalty"
    SIMILARITY_STACK = "similarity"
    RELEVANCE_STACK = "relevance"


@dataclass
class Task(ABC):
    # topics: dict
    name: str
    desc: str
    goal: str
    query: str
    topic: str
    subtopic: str
    tags: List[str]
    reward_definition = List[dict]
    reward_threshold: float = 0.0
    reference: Union[str, List[str]] = None
    criteria: str = ("",)
    delimiter: str = ""
    complete: bool = False
    static_reference: bool = False
    static_query: bool = False
    reference_system_prompt = ""
    reference_prompt = ""
    query_system_prompt = ""
    query_prompt = ""

    def __post_init__(self):
        self.cleaner = GenerationCleaner()

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, desc={self.desc!r}, goal={self.goal!r}, query={self.query!r}, reference={self.reference!r}, topic={self.topic!r}, subtopic={self.subtopic!r}, tags={self.tags!r})"

    def __repr__(self):
        return str(self)

    def __state_dict__(self, full=False):
        state = {
            "task": self.name,
            "desc": self.desc,
            "goal": self.goal,
            "query": self.query,  # For now we just use the raw query but should add delimiters again
            "query_time": getattr(self, "query_time", 0),
            "reference": self.reference,
            "reference_time": getattr(self, "reference_time", 0),
            "topic": self.topic,
            "subtopic": self.subtopic,
            "context_time": self.context.get("fetch_time", 0.0),
            # "tags": self.tags,
        }
        if full:
            state.update(**self.context)

        return state

    def generate(self, system: str, prompt: str, llm: Pipeline) -> str:
        """Uses the llm to generate a response to a prompt"""

        generation = HuggingFaceLLM(llm, system_prompt=system).query(prompt)
        generation = self.cleaner.apply(generation=generation, task_name=self.name)
        return generation

    def generate_reference(self, llm) -> str:
        """Generates a reference answer to be used for scoring miner completions"""
        t0 = time.time()
        if not self.static_reference:
            bt.logging.info("🤖 Generating reference...")

            self.reference = self.generate(
                system=self.reference_system_prompt,
                prompt=self.reference_prompt,
                llm=llm,
            )

        self.reference_time = time.time() - t0
        return self.reference

    def generate_query(self, llm) -> str:
        """Generates a query to be used for generating the challenge"""
        t0 = time.time()
        if not self.static_query:
            bt.logging.info("🤖 Generating query...")

            self.query = self.generate(
                system=self.query_system_prompt, prompt=self.query_prompt, llm=llm
            )

        self.query_time = time.time() - t0
        return self.query

    def format_challenge(self, challenge) -> str:
        """Formats the challenge to be used for the conversation"""
        challenge = self.cleaner.apply(generation=challenge, task_name=self.name)

        return challenge
