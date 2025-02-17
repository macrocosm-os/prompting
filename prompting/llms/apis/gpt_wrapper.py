import numpy as np
import openai
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from prompting.llms.apis.llm_messages import LLMMessage, LLMMessages
from shared import settings

shared_settings = settings.shared_settings


class GPT(BaseModel):
    client: openai.Client | None = None
    async_client: openai.AsyncClient | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, api_key: str = shared_settings.OPENAI_API_KEY):
        super().__init__()
        if api_key:
            self.client = openai.Client(api_key=api_key)
            self.async_client = openai.AsyncClient(api_key=api_key)

    def test(self):
        try:
            response, _ = self.chat_complete(
                messages=LLMMessages(
                    LLMMessage(
                        role="user",
                        content="Respond back saying only the word 'Hello'",
                    )
                ),
                model="gpt-3.5-turbo",
            )
            assert response.choices[0].message.content.lower() == "hello"
        except Exception as ex:
            logger.exception(f"Failed GPT test: {ex}")

    def chat_complete(
        self,
        messages: LLMMessages,
        model: str = "gpt-3.5-turbo",
        tools=None,
        retries: int = 3,
        n=1,
        max_tokens: int | None = None,
        min_tokens: int | None = 0,
        **kwargs,
    ) -> tuple[ChatCompletion, float]:
        """
        Completes a chat conversation using the OpenAI GPT model.

        Args:
            messages (GPTMessages): The chat conversation messages.
            model (str, optional): The GPT model to use. Defaults to "gpt-3.5-turbo".
            tools (list[ProbabilisticModel] | None, optional): The probabilistic models to use. Defaults to None.
            retries (int, optional): The number of retries if the GPT call fails. Defaults to 3.
            n (int, optional): The number of completions to generate. Defaults to 1.
            max_tokens (int | None, optional): The maximum number of tokens in the generated completion.
            min_tokens (int | None, optional): The minimum number of tokens in the generated completion. Defaults to 0.
            **kwargs: Additional keyword arguments to pass to the GPT API.

        Returns:
            tuple[ChatCompletion, float]: A tuple containing the chat
                                        completion response and the cost of the completion.

        Raises:
            Exception: If the GPT call fails after the specified number of retries.
        """
        tools = [tool.dump_tool for tool in tools] if tools else None
        input_tokens = messages.get_tokens(model=model)

        # TODO: Later actually calculate how many tokens are needed for the tools rather than using
        # 1k tokens as a placeholder
        tool_tokens = 1000 if tools else 0
        while True:
            # If no max tokens are specified, we use however many tokens are left in the context window
            output_tokens = min(
                shared_settings.GPT_MODEL_CONFIG[model]["context_window"] - input_tokens - tool_tokens,
                shared_settings.GPT_MODEL_CONFIG[model]["max_tokens"],
            )
            if min_tokens < output_tokens:
                break
            else:
                model = shared_settings.GPT_MODEL_CONFIG[model].get("upgrade")
                if model is None:
                    raise ValueError(
                        f"Minimum tokens ({min_tokens}) exceed the available output tokens ({output_tokens})"
                    )

        if max_tokens is not None:
            output_tokens = min(output_tokens, max_tokens)

        for _ in range(retries):
            try:
                msgs = messages.to_dict()
                response: ChatCompletion = self.client.chat.completions.create(
                    messages=msgs,
                    model=model,
                    tools=tools,
                    n=n,
                    max_tokens=output_tokens,
                    **kwargs,
                )
                output_cost = (
                    response.usage.completion_tokens * shared_settings.GPT_MODEL_CONFIG[model]["output_token_cost"]
                ) / 1000
                input_cost = (
                    response.usage.prompt_tokens * shared_settings.GPT_MODEL_CONFIG[model]["input_token_cost"]
                ) / 1000
                return response, output_cost + input_cost
            except Exception as ex:
                logger.exception(f"GPT call failed: {ex}")
        raise Exception(f"GPT call failed after {retries} retries")

    async def chat_complete_async(
        self,
        messages: LLMMessages,
        model: str = "gpt-3.5-turbo",
        tools=None,
        retries: int = 3,
        n=1,
        max_tokens: int | None = None,
        min_tokens: int | None = 0,
        **kwargs,
    ) -> tuple[ChatCompletion, float]:
        """
        Completes a chat conversation using the OpenAI GPT model.

        Args:
            messages (GPTMessages): The chat conversation messages.
            model (str, optional): The GPT model to use. Defaults to "gpt-3.5-turbo".
            tools (list[ProbabilisticModel] | None, optional): The probabilistic models to use. Defaults to None.
            retries (int, optional): The number of retries if the GPT call fails. Defaults to 3.
            n (int, optional): The number of completions to generate. Defaults to 1.
            max_tokens (int | None, optional): The maximum number of tokens in the generated completion.
            min_tokens (int | None, optional): The minimum number of tokens in the generated completion. Defaults to 0.
            **kwargs: Additional keyword arguments to pass to the GPT API.

        Returns:
            tuple[ChatCompletion, float]: A tuple containing the chat
                                        completion response and the cost of the completion.

        Raises:
            Exception: If the GPT call fails after the specified number of retries.
        """
        tools = [tool.dump_tool for tool in tools] if tools else None
        input_tokens = messages.get_tokens(model=model)

        while True:
            # If no max tokens are specified, we use however many tokens are left in the context window
            output_tokens = min(
                shared_settings.GPT_MODEL_CONFIG[model]["context_window"] - input_tokens,
                shared_settings.GPT_MODEL_CONFIG[model]["max_tokens"],
            )
            if min_tokens < output_tokens:
                break
            else:
                model = shared_settings.GPT_MODEL_CONFIG[model].get("upgrade")
                if model is None:
                    raise ValueError(
                        f"Minimum tokens ({min_tokens}) exceed the available output tokens ({output_tokens})"
                    )

        if max_tokens is not None:
            output_tokens = min(output_tokens, max_tokens)

        for _ in range(retries):
            try:
                msgs = messages.to_dict()
                response: ChatCompletion = await self.async_client.chat.completions.create(
                    messages=msgs,
                    model=model,
                    tools=tools,
                    n=n,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                output_cost = (
                    response.usage.completion_tokens * shared_settings.GPT_MODEL_CONFIG[model]["output_token_cost"]
                ) / 1000
                input_cost = (
                    response.usage.prompt_tokens * shared_settings.GPT_MODEL_CONFIG[model]["input_token_cost"]
                ) / 1000
                return response, output_cost + input_cost
            except Exception as ex:
                logger.exception(f"GPT call failed: {ex}")
        raise Exception(f"GPT call failed after {retries} retries")

    def get_embeddings(
        self,
        text: list[str],
        model: str = "text-embedding-3-small",
        retries: int = 3,
        **kwargs,
    ) -> list[np.ndarray]:
        assert isinstance(text, list), "Text must be a list of strings"
        for i in range(retries):
            try:
                embeddings = self.client.embeddings.create(input=text, model=model, **kwargs)
                return [np.array(embedding.embedding) for embedding in embeddings.data]
            except Exception as ex:
                logger.error(f"GPT embedding call {i} failed: {ex}\n\nInputs:{text}")
        raise Exception(f"GPT embedding call failed after {retries} retries")


openai_client = GPT(api_key=shared_settings.OPENAI_API_KEY)
