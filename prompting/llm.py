import random
import textwrap

import bittensor as bt
import textwrap
from typing import List
from dataclasses import dataclass, asdict
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import Pipeline
import time
from abc import ABC, abstractmethod



class LLM(ABC):
    def __init__(self, pipeline, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95):
        self.pipeline = pipeline
        self.kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        self.messages = []
        self.times = []

    @abstractmethod
    def query(self, message, cleanup=True, role="user"):
        pass

    @abstractmethod
    def forward(self, messages, cleanup=False):
        pass

    def _make_prompt(self, messages):
        # Abstract method for creating the prompt from messages
        pass
    
    def __call__(self, messages):
        return self.forward(messages)





class HuggingFaceLLM(LLM):

    def __init__(
        self,
        pipeline: Pipeline,
        system_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    ):
        self.pipeline = pipeline
        self.system_prompt = system_prompt
        self.kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        self.messages = [{"content": self.system_prompt, "role": "system"}]
        self.times = [0]

    def query(self, message, cleanup=True, role="user", disregard_system_prompt=False):
        messages = self.messages + [{"content": message, "role": role}]

        if disregard_system_prompt:
            messages = messages[1:]

        tbeg = time.time()
        response = self.forward(messages, cleanup=cleanup)

        self.messages = messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - tbeg]
        return response

    def __call__(self, messages):
        return self.forward(messages)

    def _make_prompt(self, messages):
        return self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def forward(self, messages, cleanup=False, preformat_messages=False):
        prompt = self._make_prompt(messages)
        outputs = self.pipeline(prompt, **self.kwargs)
        response = outputs[0]["generated_text"]

        response = response.replace(prompt, "").strip()
        response.split("\n")
        if cleanup and response.startswith("Assistant:"):
            print(f"Cleaning up response: {response}")
            response = response.strip("Assistant:").split("User:")[0].strip("\n")

        return response


################### OpenAI LLM ###################

# To have an OpenAI based flow, replace the LLM class with this one by commenting the code above and uncommenting the code below. You will need a dotenv file to hold your OpenAI API key.
import openai
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

class OpenAILLM:
    # default_prompt = "You are a straight-to-the-point AI assistant who always responds concisely. You are honest about things you don't know. Do not greet the user or add any extra personality or flair"
    default_prompt = "You are a question creating AI assistant. You try to make concise questions that adhere to your given instructions"

    def __init__(
        self,
        pipeline: Pipeline,
        system_prompt=None,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    ):
        self.pipeline = pipeline
        if system_prompt is None:
            system_prompt = self.default_prompt
        self.system_prompt = system_prompt


        self.messages = [{"content": self.system_prompt, "role": "system"}]
        self.times = [0]

        _ = load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']

        self.client = OpenAI()

    def query(self, message, cleanup=True, role="user", disregard_system_prompt=False):
        messages = self.messages + [{"content": message, "role": role}]

        # if disregard_system_prompt:
        #     messages = messages[1:]

        tbeg = time.time()
        response = self.forward(messages, cleanup=cleanup)

        self.messages = messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - tbeg]
        return response


    def _make_prompt(self, messages):
        # Concatenate the messages to form the prompt
        prompt = ""
        for message in messages:
            prompt += f"{message['role'].capitalize()}: {message['content']}\n"
        return prompt


    def forward(self, messages, cleanup=False, preformat_messages=False):
        outputs = self.client.chat.completions.create(
            model="gpt-4",
            messages = messages,
        )
        response = outputs.choices[0].message.content

        return response
