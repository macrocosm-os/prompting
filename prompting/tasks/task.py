import time
import bittensor as bt
from abc import ABC
from pydantic import BaseModel
from typing import Union
from prompting.llms import vLLM_LLM, BasePipeline
from prompting.cleaners.cleaner import CleanerPipeline
from prompting.rewards import BaseRewardModel


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


class Task(ABC, BaseModel):
    # topics: dict
    name: str
    desc: str
    goal: str
    query: str
    topic: str
    subtopic: str
    tags: list[str]
    context: dict
    reward_definition: list[BaseRewardModel]
    penalty_definition: list[BaseRewardModel] = None
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

    global_penalty_definition = [dict(name="streaming", max_tokens_per_chunk=200, weight=0.2)]

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, desc={self.desc!r}, goal={self.goal!r}, query={self.query!r}, reference={self.reference!r}, topic={self.topic!r}, subtopic={self.subtopic!r}, tags={self.tags!r})"

    def __repr__(self):
        return str(self)

    # def __state_dict__(self, full=False):
    #     state = {
    #         "task": self.name,
    #         "desc": self.desc,
    #         "goal": self.goal,
    #         "query": self.query,  # For now we just use the raw query but should add delimiters again
    #         "query_time": getattr(self, "query_time", 0),
    #         "reference": self.reference,
    #         "reference_time": getattr(self, "reference_time", 0),
    #         "topic": self.topic,
    #         "subtopic": self.subtopic,
    #         "context_time": self.context.stats.get("fetch_time", 0.0),
    #     }
    #     if full:
    #         state.update(asdict(self.context))

    #     return state

    def generate(self, system: str, prompt: str, pipeline: BasePipeline, clean=True) -> str:
        """Uses the llm to generate a response to a prompt"""
        return vLLM_LLM(pipeline, system_prompt=system).query(message=prompt, cleaner=self.cleaner)

    def generate_reference(self, pipeline: BasePipeline, clean=True) -> str:
        """Generates a reference answer to be used for scoring miner completions"""
        t0 = time.time()
        if not self.static_reference:
            bt.logging.info("🤖 Generating reference...")
            self.reference = self.generate(
                system=CHATTENSOR_SYSTEM_PROMPT(),
                prompt=self.reference_prompt,
                pipeline=pipeline,
                clean=self.clean_reference,
            )

        self.reference_time = time.time() - t0
        return self.reference

    def generate_query(self, pipeline: BasePipeline, clean=True) -> str:
        """Generates a query to be used for generating the challenge"""
        t0 = time.time()
        if not self.static_query:
            bt.logging.info("🤖 Generating query...")
            self.query = self.generate(
                system=self.query_system_prompt,  # Could possibly add the chattensor system prompt to query but I don't think it adds anything
                prompt=self.query_prompt,
                pipeline=pipeline,
                clean=clean,
            )

        self.query_time = time.time() - t0
        return self.query
