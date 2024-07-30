import time
import bittensor as bt
from abc import ABC
from pydantic import BaseModel
from typing import Union
from prompting.llms.base_llm import BasePipeline
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.cleaners.cleaner import CleanerPipeline
from prompting.rewards.reward import BaseRewardModel, RewardEvent
from pydantic import model_validator
from prompting.dendrite import DendriteResponseEvent
import numpy as np


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


class WeightedRewardModel(BaseModel):
    weight: float
    reward_model: BaseRewardModel


class WeightedRewardEvent(BaseModel):
    weight: float
    reward_event: RewardEvent


class BaseRewardConfig(ABC, BaseModel):
    """This class takes in a dictionary of rewards and penalties that should be applied. On apply(),
    it then applies all the reward models based on query & reference and returns the reward.

    both reward_definition and penalty_definition must be a list of tuples of type:

    weighting: RewardModel, e.g.

    [ (0.2, RougeRewardModel), (0.8, CosineDistanceRewardModel) ]

    Note that for all the rewards, the percentages must sum up to 1 (100%). For penalties,
    this is not the case, e.g. you may want to only apply a single penalty very lightly
    and weight it with <1.
    """

    reward_definitions: list[WeightedRewardModel]
    penalty_definitions: list[WeightedRewardModel] = []

    reward_events: list[WeightedRewardEvent] | None = None
    penalty_events: list[WeightedRewardEvent] | None = None

    @property
    def total_reward(self) -> list[float]:
        if not self.reward_events:
            raise Exception("Rewards have not yet been calculated")
        return np.sum([r.reward_event.rewards for r in self.reward_events], axis=0)

    @property
    def total_penalty(self) -> list[float]:
        if not self.penalty_events:
            return 0
        return np.sum([r.reward_event.rewards for r in self.penalty_events], axis=0)

    @property
    def final_reward(self) -> list[float]:
        return self.total_reward - self.total_penalty

    @model_validator(mode="after")
    def check_summation(self) -> "BaseRewardConfig":
        assert sum([r.weight for r in self.reward_definitions]) == 1, "All rewards must sum to one"

    def apply(self, reference, response_event: DendriteResponseEvent):
        for weighted_reward in self.reward_definitions:
            self.reward_events = []
            self.reward_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=reference, response_event=response_event, reward_type="reward"
                    ),
                )
            )

        for weighted_reward in self.penalty_definitions:
            self.penalty_events = []
            self.penalty_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=reference, response_event=response_event, reward_type="penalty"
                    ),
                )
            )
        return self.final_reward


class Task(ABC, BaseModel):
    context: dict
    reward_config: BaseRewardConfig

    query: str | None = None
    reward_threshold: float = 0.0
    reference: Union[str, list[str]] = ""
    criteria: str = ("",)
    delimiter: str = ""
    complete: bool = False
    static_reference: bool = False
    static_query: bool = False
    reference_prompt: str = ""
    query_system_prompt: str = ""
    query_prompt: str = ""
    cleaner: CleanerPipeline = CleanerPipeline()
    clean_reference: bool = True
    challenge_type: str = "inference"
    query_time: int | None = None
    reference_time: int | None = None

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, desc={self.desc!r}, goal={self.goal!r}, query={self.query!r}, reference={self.reference!r}, topic={self.topic!r}, subtopic={self.subtopic!r}, tags={self.tags!r})"

    def __repr__(self):
        return str(self)

    def generate(self, system: str, prompt: str, pipeline: BasePipeline, clean=True) -> str:
        """Uses the llm to generate a response to a prompt"""
        return vLLM_LLM(pipeline, system_prompt=system).query(message=prompt, cleaner=self.cleaner)

    def generate_reference(self, pipeline: BasePipeline, reference_prompt: str, clean=True) -> str:
        """Generates a reference answer to be used for scoring miner completions"""
        t0 = time.time()
        if not self.static_reference:
            bt.logging.info("🤖 Generating reference...")
            self.reference = self.generate(
                system=CHATTENSOR_SYSTEM_PROMPT(),
                prompt=reference_prompt,
                pipeline=pipeline,
                clean=clean,
            )

        self.reference_time = time.time() - t0
        return self.reference

    def generate_query(
        self, pipeline: BasePipeline, query_prompt: str, query_system_prompt: str | None = None, clean=True
    ) -> str:
        """Generates a query to be used for generating the challenge"""
        t0 = time.time()
        if not self.static_query:
            bt.logging.info("🤖 Generating query...")
            self.query = self.generate(
                system=query_system_prompt,  # Could possibly add the chattensor system prompt to query but I don't think it adds anything
                prompt=query_prompt,
                pipeline=pipeline,
                clean=clean,
            )

        self.query_time = time.time() - t0
        return self.query
