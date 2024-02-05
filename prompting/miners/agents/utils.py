import torch
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool
from langchain.llms.huggingface_pipeline import HuggingFacePipeline


def get_tools():
    wikipedia = WikipediaAPIWrapper()
    tools = [
        Tool(
            name="wikipedia",
            func=wikipedia.run,
            description="Useful for when you need to look up a topic, country or person on wikipedia",
        )
    ]

    return tools


def load_hf_llm(model_id:str, max_new_tokens:int, load_in_8bits: bool ,load_in_4bits: bool):
    model_kwargs = { "torch_dtype": torch.float16 }

    if load_in_8bits:         
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bits:
        model_kwargs["load_in_4bit"] = True

    
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",        
        device_map="auto",        
        pipeline_kwargs={"max_new_tokens": max_new_tokens},
        model_kwargs=model_kwargs
    )

    return llm

