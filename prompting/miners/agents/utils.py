import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline


def load_hf_llm(
    model_id: str, max_new_tokens: int, load_in_8bits: bool, load_in_4bits: bool
):
    model_kwargs = {"torch_dtype": torch.float16}

    if load_in_8bits:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bits:
        model_kwargs["load_in_4bit"] = True

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        # TODO: Add device from config dynamically
        device=0,
        pipeline_kwargs={"max_new_tokens": max_new_tokens},
        model_kwargs=model_kwargs,
    )

    return llm
