from typing import Literal
from PIL import Image
from pydantic import BaseModel
from prompting.llms.apis.gpt_wrapper import encode_image_from_memory, calculate_image_tokens, get_text_tokens


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = None
    image: Image.Image | None = None

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        if not self.image:
            content = self.content
        else:
            content = [
                {
                    "type": "text",
                    "text": self.content,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_from_memory(self.image)}",
                    },
                },
            ]

        return {
            "role": self.role,
            "content": content,
        }

    def get_tokens(self, model: str) -> int:
        total_tokens = 4  # Each message has 4 tokens (for role etc.)
        if self.image:
            total_tokens += calculate_image_tokens(self.image.width, self.image.height)
        if self.content:
            total_tokens += get_text_tokens(self.content, model)
        return total_tokens

    def __str__(self):
        return f"ROLE: {self.role}\nCONTENT:\n{self.content}\nHAS_IMAGE: {bool(self.image)}"


class LLMMessages(BaseModel):
    messages: list[LLMMessage] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.messages = list(args)
        assert len(self.messages) > 0, "At least one message is required when initializing GPTMessages"

    # TODO: Add some utility for token tracking etc.

    def to_dict(self) -> list[dict]:
        return [message.to_dict() for message in self.messages]

    def get_tokens(self, model: str) -> list[str]:
        total_tokens = 3  # (For some reason the first message has 3 additional tokens)
        for msg in self.messages:
            total_tokens += msg.get_tokens(model=model)
        return total_tokens
