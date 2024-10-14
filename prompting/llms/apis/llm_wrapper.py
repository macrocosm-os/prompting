from prompting.llms.apis.sn19_wrapper import chat_complete
from prompting.llms.apis.gpt_wrapper import openai_client
from prompting.llms.apis.llm_messages import LLMMessages
from loguru import logger


class LLMWrapper:
    def chat_complete(
        messages: LLMMessages,
        model="chat-llama-3-1-70b",
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        stream=False,
        logprobs=True,
    ):
        if "gpt" in model.lower():
            try:
                response, _ = openai_client.chat_complete(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=stream,
                    logprobs=logprobs,
                )
                return response.choices[0].message.content
            except Exception as ex:
                logger.exception(ex)
                model = "chat-llama-3-1-70b"
        return chat_complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            logprobs=logprobs,
        )
