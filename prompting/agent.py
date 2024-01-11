import random
import textwrap
import bittensor as bt
import textwrap
from tasks import Task
from dataclasses import dataclass
from llm import HuggingFaceLLM, OpenAILLM
from transformers import pipeline
import torch

@dataclass
class Persona:
    profile: str
    mood: str
    tone: str


def create_persona() -> Persona:
    """Defines the persona of the user. This is used to create the system prompt. It dictates the style of the agent's questions and communication."""
    profiles = ["student", "teacher", "parent", "hacker", "programmer", "scientist"]
    # profiles = ["16 year old highschool student", ...

    # TODO: more terse, less verbose
    mood = [
        "an interested",
        "a concerned",
        "an impatient",
        "a tired",
        "a confused",
        "an annoyed",
        "a curious",
        "an upbeat",
        "a lazy",
    ]
    tone = [
        "formal",
        "informal",
        "indifferent",
        "casual",
        "rushed",
        "polite",
        "impolite",
        "friendly",
        "unfriendly",
        "positive",
        "negative",
    ]
    # TODO: we can lower case the human messages, add common grammar and spelling mistakes...

    return Persona(
        profile=random.choice(profiles),
        mood=random.choice(mood),
        tone=random.choice(tone),
    )

class HumanAgent(HuggingFaceLLM):
    "Agent that impersonates a human user and makes queries based on its goal."

    @property
    def progress(self):
        return int(self.task.complete)

    @property
    def finished(self):
        return self.progress == 1

    system_prompt_template = textwrap.dedent(
        """This is a roleplaying game where you are impersonating {mood} human user with a specific persona. As a human, you are using AI assistant to {desc} related to {topic} ({subtopic}) in a {tone} tone. You don't need to greet the assistant or be polite, unless this is part of your persona. The spelling and grammar of your messages should also reflect your persona.

        Your singular focus is to use the assistant to {goal}: {query}
    """
    )

    def __init__(
        self,
        task: Task,
        llm,
        system_template: str = None,
        persona: Persona = None,
        begin_conversation=True,
    ):
        if persona is None:
            self.persona = create_persona()

        self.task = task
        self.llm = llm

        if system_template is not None:
            self.system_prompt_template = system_template

        self.system_prompt = self.system_prompt_template.format(
            mood=self.persona.mood,
            tone=self.persona.tone,
            **self.task.__state_dict__(),  # Adds desc, subject, topic
        )

        super().__init__(
            pipeline=llm,
            system_prompt=self.system_prompt,
            max_new_tokens=256,
        )

        if begin_conversation:
            bt.logging.info("🤖 Generating challenge query...")
            # initiates the conversation with the miner
            self.challenge = self.create_challenge()

    def create_challenge(self) -> str:
        """Creates the opening question of the conversation which is based on the task query but dressed in the persona of the user."""
        self.challenge = super().query(message="Ask a question related to your goal")
        return self.challenge


    def __str__(self):
        return self.system_prompt

    def __repr__(self):
        return str(self)

    def continue_conversation(self, miner_response: str):
        # Generates response to miner response
        self.query(miner_response)
        # Updates current prompt with new state of conversation
        self.prompt = self.get_history_prompt()

    def update_progress(
        self, top_reward: float, top_response: str, continue_conversation=False
    ):
        if top_reward > self.task.reward_threshold:
            self.task.complete = True
            self.messages.append({"content": top_response, "role": "user"})

            bt.logging.info("Agent finished its goal")
            return

        if continue_conversation:
            bt.logging.info(
                "↪ Agent did not finish its goal, continuing conversation..."
            )
            self.continue_conversation(miner_response=top_response)


# TEST AGENT
if __name__ == "__main__":
    bt.logging.info("🤖 Loading LLM model...")

    llm_pipeline = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        #device_map="cuda:0",
        device_map="auto",
    )

    # bt.logging.info("Creating task...")
    # dataset = WikiDataset()
    # context = dataset.next()

    # task = SummarizationTask(llm_pipeline=llm_pipeline, context=context)    

    # bt.logging.info("Creating agent...")
    # agent = HumanAgent(
    #     task = task,
    #     # TODO: better design dependency: Agent contains an LLM instead of being one, this will enable us to use different LLMs without changing the agent
    #     llm=llm_pipeline,
    #     system_template=None,
    #     begin_conversation=True
    # )

    # bt.logging.info(f'Agent created with persona: {agent.persona}')
    # bt.logging.info(f'Task query: {task.query}')
    # bt.logging.info(f'Task reference: {task.reference}')
    # bt.logging.info(f'Agent challenge: {agent.challenge}')



