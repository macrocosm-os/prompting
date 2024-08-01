import time
import textwrap
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
from prompting.persona import Persona


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
    def total_rewards(self) -> list[float]:
        if not self.reward_events:
            raise Exception("Rewards have not yet been calculated")
        return np.sum([r.reward_event.rewards for r in self.reward_events], axis=0)

    @property
    def total_penalties(self) -> list[float]:
        if not self.penalty_events:
            return 0
        return np.sum([r.reward_event.rewards for r in self.penalty_events], axis=0)

    @property
    def final_rewards(self) -> list[float]:
        return self.total_rewards - self.total_penalties

    @model_validator(mode="after")
    def check_summation(self) -> "BaseRewardConfig":
        assert sum([r.weight for r in self.reward_definitions]) == 1, "All rewards must sum to one"

    def apply(self, response_event: DendriteResponseEvent, reference: str, challenge: str) -> list[float]:
        for weighted_reward in self.reward_definitions:
            self.reward_events = []
            self.reward_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=reference, response_event=response_event, challenge=challenge, reward_type="reward"
                    ),
                )
            )

        for weighted_reward in self.penalty_definitions:
            self.penalty_events = []
            self.penalty_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=challenge, response_event=response_event, reward_type="penalty"
                    ),
                )
            )
        return self.final_rewards


class BaseTask(ABC, BaseModel):
    context: dict
    augment: bool = False

    query: str | None = None
    augmented_query: str | None = None

    reward_threshold: float = 0.0
    reference: Union[str, list[str]] = ""
    criteria: str = ("",)
    delimiter: str = ""
    complete: bool = False
    reference_prompt: str = ""
    query_system_prompt: str = ""
    query_prompt: str = ""
    cleaner: CleanerPipeline = CleanerPipeline()
    clean_reference: bool = True
    challenge_type: str = "inference"
    query_time: int | None = None
    reference_time: int | None = None

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.__class__.__name__!r}, query={self.query!r}, reference={self.reference!r})"

    def __repr__(self):
        return str(self)

    def generate_reference(self, pipeline: BasePipeline) -> str:
        """Generates a reference answer to be used for scoring miner completions"""
        if len(self.reference_prompt) == 0:
            bt.logging.error("Reference prompt is empty. Please provide a reference prompt.")

        bt.logging.info("🤖 Generating reference...")
        self.reference = vLLM_LLM(pipeline, system_prompt=CHATTENSOR_SYSTEM_PROMPT()).query(
            message=self.reference_prompt, cleaner=self.cleaner
        )
        return self.reference

    def generate_query(
        self,
        pipeline: BasePipeline,
        persona: Persona = Persona(),
    ) -> str:
        """Generates a query to be used for generating the challenge"""
        bt.logging.info("🤖 Generating query...")
        self.query = vLLM_LLM(pipeline, system_prompt=self._system_prompt_template(persona=persona)).query(
            message=self.query_prompt, cleaner=self.cleaner
        )

        self.augmented_query = self.augment_query(llm_pipeline=pipeline, persona=persona)
        return self.query

    def _system_prompt_template(self, persona: Persona) -> str:
        return textwrap.dedent(
            f"""This is a roleplaying game where you are impersonating a {persona.mood} human user who is using an AI to help you with a task.

            The task is: {self.query}

            Rephrase this query in a {persona.tone} tone, and ask the AI to help you with it. Be creative and have fun with testing the AI!
        """
        )

    def augment_query(
        self,
        llm_pipeline: BasePipeline,
        persona: Persona,
    ) -> str:
        """Creates the opening question of the conversation which is based on the task query but dressed in the persona of the user."""
        if not self.augment:
            return self.query

        if self.challenge_type == "inference":
            challenge = vLLM_LLM(
                llm_pipeline=llm_pipeline, max_new_tokens=256, system_prompt=self._system_prompt_template(persona)
            )
        elif self.challenge_type == "query":
            challenge = self.query
        else:
            bt.logging.error(
                f"Task {self.__class__.__name__} has challenge type of: {self.challenge_type} which is not supported."
            )

        return challenge
