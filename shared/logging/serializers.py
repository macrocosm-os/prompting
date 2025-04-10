from logging import logger

import numpy as np
from openai.types.chat import ChatCompletionChunk

from shared.logging.serializer_registry import register_serializer


@register_serializer(ChatCompletionChunk)
def serialize_chunk(chunk: ChatCompletionChunk) -> dict:
    # Currently only serializes the content and logprobs of the first choice
    try:
        delta = chunk.choices[0].delta
        return {
            "content": getattr(delta, "content", None),
            "logprobs": getattr(chunk.choices[0], "logprobs", None),
        }
    except Exception as e:
        logger.error(f"Error serializing chunk: {chunk}")
        logger.error(f"Error: {e}")
        return {"content": None, "logprobs": None}


@register_serializer(np.ndarray)
def serialize_ndarray(array: np.ndarray) -> list:
    return array.tolist()


@register_serializer(np.float64)
def serialize_float64(value: np.float64) -> float:
    return float(value)
