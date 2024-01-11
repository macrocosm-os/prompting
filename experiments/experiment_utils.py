## Utility functions for various notebooks...
import torch
import bittensor as bt

import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

OUTPUT_PARSER = StrOutputParser()
OPENAI_API_KEY = "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"


#### PROMETHEUS RUBRIC SCORING METRIC METHODOLOGY ####
def prometheus_rubric():
    system_prompt = """\
    You are a student in a class. Your teacher has given you a homework assignment. Your task is to provide a response to the query which is of the specified quality (you can think of this as the grade). You will be provided with a query, a reference and a target score.

    The quality system is a score from 1 to 3, where 1 is bad, 2 is okay, and 3 is great response. A reference answer is provided which by definition should score the maximum of 3. A rubric is also provided which describes the quality of the answer at each level.

    # Rubric
    1. The response is completely unhelpful. It is irrelevant, uninformative, potentially misdirecting or untruthful or it is a copy of the query. It may contain illogical or incoherent statements. It may also try to change the subject or be a non-sequitur. Reply with the query or a non-sequitur, or complete gibberish.
    2. The response is relevant and informative but contains factual inaccuracies or is incomplete. It may also contain some irrelevant information.
    3. By all measures the response is as good as the reference answer. It is relevant, informative, complete and accurate. It does not contain irrelevant information, nor does it contain any factual inaccuracies.

    """

    user_prompt_template = """\
    # Query
    {query}

    # Reference Answer (scores 3)
    {reference}

    Produce a response which is characteristic of a score of {desired_score}.
    """

    return {
        "possible_scores": [1, 2, 3],
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
    }


def load_llm(model: str, **kwargs):
    bt.logging.info(f"🤖 Loading LLM model {model}...")
    if model == "zephyr":
        llm = HuggingFacePipeline.from_model_id(
            model_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            device=0,  # replace with device_map="auto" to use the accelerate library.
            # device_map="cuda:0",
            pipeline_kwargs={"max_new_tokens": 256},
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
    elif model.startswith("gpt"):
        llm = ChatOpenAI(
            model_name=model, max_tokens=256, api_key=OPENAI_API_KEY, **kwargs
        )
    else:
        raise NotImplementedError(f"Model {model} not implemented")

    bt.logging.success(f"🤖 Loaded LLM model {model}!")
    return llm


def get_gpt_reference(message, model, output_parser):
    prompt = ChatPromptTemplate.from_messages([("user", "{input}")])

    chain = prompt | model | output_parser
    response = chain.invoke({"input": message})

    return response
