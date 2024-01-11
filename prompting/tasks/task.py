import bittensor as bt
from abc import ABC
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Union
from llm import HuggingFaceLLM, OpenAILLM


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
    criteria: str = "",
    delimiter: str = ""
    complete: bool = False
    static_reference: bool = False
    static_query: bool = False

    def __str__(self):
        return f"{self.desc} about {self.topic} ({self.subject})"

    def __repr__(self):
        return f"{self.desc} about {self.topic} ({self.subject})"

    def asdict(self):
        return {
            k: v
            for k, v in asdict(self).items()
            if k in ("topic", "subtopic", "reference_prompt", "reference_system_prompt", "goal", "desc", "name")
        }

    def __state_dict__(self):
        return {
            "desc": self.desc,
            "goal": self.goal,
            "query": self.query, # For now we just use the raw query but should add delimiters again
            "topic": self.topic,
            "subtopic": self.subtopic,
        }

    def formatted_reference(self):
        return f"{self.delimiter}{self.reference}{self.delimiter}"

    def formatted_challenge(self):
        return f"{self.delimiter}{self.challenge}{self.delimiter}"


    def generate_reference(self, llm):
        """Generates a reference answer to be used for scoring miner completions"""
        if self.static_reference:
            return self.reference
        bt.logging.info("🤖 Generating reference...")
        self.reference = self.generate(
            system=self.reference_system_prompt,
            prompt=self.reference_prompt,
            llm=llm
        )
        return self.reference

    def generate_query(self, llm):
        """Generates a query to be used for generating the challenge"""
        bt.logging.info("🤖 Generating query...")
        if self.static_query:
            return self.query
        self.query = self.generate(
            system=self.query_system_prompt,
            prompt=self.query_prompt,
            llm=llm
        )
        return self.query

    def generate(self, system, prompt, llm=None):
        if llm is None:
            raise ValueError("OPENAI IS Broken")
        else:
            agent = HuggingFaceLLM(llm, system_prompt=system)

        return agent.query(prompt)


