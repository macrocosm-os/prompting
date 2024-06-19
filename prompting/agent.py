# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import textwrap
import time
import bittensor as bt
from dataclasses import asdict
from prompting.tasks import Task
from prompting.llms import HuggingFaceLLM, vLLM_LLM
from prompting.cleaners.cleaner import CleanerPipeline

from prompting.persona import Persona, create_persona

from transformers import Pipeline


class HumanAgent(vLLM_LLM):
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
        llm_pipeline: Pipeline,
        system_template: str = None,
        persona: Persona = None,
        begin_conversation=True,
    ):
        if persona is None:
            persona = create_persona()

        self.persona = persona
        self.task = task
        self.llm_pipeline = llm_pipeline

        if system_template is not None:
            self.system_prompt_template = system_template

        self.system_prompt = self.system_prompt_template.format(
            mood=self.persona.mood,
            tone=self.persona.tone,
            **self.task.__state_dict__(),  # Adds desc, subject, topic
        )

        super().__init__(
            llm_pipeline=llm_pipeline,
            system_prompt=self.system_prompt,
            max_new_tokens=256,
        )

        if begin_conversation:
            bt.logging.info("🤖 Generating challenge query...")
            # initiates the conversation with the miner
            self.challenge = self.create_challenge()

    def create_challenge(self) -> str:
        """Creates the opening question of the conversation which is based on the task query but dressed in the persona of the user."""
        t0 = time.time()

        cleaner = None
        if hasattr(self.task, "cleaning_pipeline"):
            cleaner = CleanerPipeline(cleaning_pipeline=self.task.cleaning_pipeline)
        if self.task.challenge_type == "inference":
            self.challenge = super().query(
                message="Ask a question related to your goal", cleaner=cleaner
            )
        elif self.task.challenge_type == 'paraphrase':
            self.challenge = self.task.challenge_template.next(self.task.query)
        elif self.task.challenge_type == 'query':
            self.challenge = self.task.query
        else:
            bt.logging.error(f"Task {self.task.name} has challenge type of: {self.task.challenge_type} which is not supported.")
        self.challenge = self.task.format_challenge(self.challenge)
        self.challenge_time = time.time() - t0

        return self.challenge

    def __state_dict__(self, full=False):
        return {
            "challenge": self.challenge,
            "challenge_time": self.challenge_time,
            **self.task.__state_dict__(full=full),
            **asdict(self.persona),
            "system_prompt": self.system_prompt,
        }

    def __str__(self):
        return self.system_prompt

    def __repr__(self):
        return str(self)

    def continue_conversation(self, miner_response: str):
        # Generates response to miner response
        self.query(miner_response)
        # Updates current prompt with new state of conversation
        # self.prompt = self.get_history_prompt()

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
