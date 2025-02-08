import json

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from prompting.llms.apis.llm_messages import LLMMessages
from shared.settings import shared_settings


# TODO: key error in response.json() when response is 500
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def chat_complete(
    messages: LLMMessages,
    model="chat-llama-3-1-70b",
    temperature=0.5,
    max_tokens=500,
    top_p=1,
    stream=False,
    logprobs=True,
):
    url = f"{shared_settings.SN19_API_URL}v1/chat/completions"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {shared_settings.SN19_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "messages": messages.to_dict(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model": model,
        "top_p": top_p,
        "stream": stream,
        "logprobs": logprobs,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
    response_json = response.json()
    try:
        return response_json["choices"][0]["message"].get("content")
    except KeyError:
        return response_json["choices"][0]["delta"].get("content")
