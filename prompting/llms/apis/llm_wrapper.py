from loguru import logger

from prompting.llms.apis.gpt_wrapper import openai_client
from prompting.llms.apis.llm_messages import LLMMessages
from prompting.llms.apis.sn19_wrapper import chat_complete
from shared.settings import shared_settings


class LLMWrapper:
    def chat_complete(
        messages: LLMMessages,
        model: str = "chat-llama-3-1-70b",
        temperature: float = 0.5,
        max_tokens: int = 500,
        top_p: float = 1,
        stream: bool = False,
        logprobs: bool = True,
    ) -> str:
        response: str | None = None
        if "gpt" not in model.lower() and shared_settings.SN19_API_KEY and shared_settings.SN19_API_URL:
            try:
                response = chat_complete(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=stream,
                    logprobs=logprobs,
                )
                logger.debug(f"Generated {len(response)} characters using {model}")
                return response

            except Exception as ex:
                logger.error(
                    "Failed to use SN19 API, falling back to GPT-3.5. "
                    f"Make sure to setup 'SN19_API_KEY' and 'SN19_API_URL' in .env.validator"
                )

        model = "gpt-3.5-turbo"
        response, _ = openai_client.chat_complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            logprobs=logprobs,
        )
        response = response.choices[0].message.content
        logger.debug(f"Generated {len(response)} characters using {model}")
        return response
