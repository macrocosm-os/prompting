from loguru import logger

try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    import torch
except ImportError:
    logger.warning("Transformers or torch is not installed. This module will not be available.")

from prompting.llms.hf_llm import ReproducibleHF

class HFTextImageToText(ReproducibleHF):
    def __init__(self, model_id: str = "google/gemma-3-27b-it", device: str = "cuda:0", sampling_params: dict[str, str | float | int | bool] | None = None):
        super().__init__(model_id, device, sampling_params)
        self.model: Gemma3ForConditionalGeneration = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        self.tokenizer = AutoProcessor.from_pretrained(model_id)
        self.valid_generation_params = set(
            self.model.generation_config.to_dict().keys()
        )
        self.message_formater = self.format_messages

    @staticmethod
    def format_messages(messages: list[str] | list[dict[str, str]]) -> list[dict[str, str | list[dict[str, str]]]]:
        """Format the messages for the gemma model.
        
        Converts message content strings to dictionaries with type and text fields.
        Example:
        Input: [{"role": "user", "content": "Hello"}]
        Output: [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        """
        formatted_messages = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                # If content is a string, convert it to a list with a dictionary
                if isinstance(message["content"], str):
                    formatted_message = message.copy()
                    formatted_message["content"] = [{"type": "text", "text": message["content"]}]
                    formatted_messages.append(formatted_message)
                else:
                    # If content is already in the correct format, keep it as is
                    formatted_messages.append(message)
            else:
                # Handle other message formats if needed
                formatted_messages.append(message)
        
        return formatted_messages
    
if __name__ == "__main__":
    model = HFTextImageToText(model_id="google/gemma-3-27b-it", device="cuda:0")
    print(model.generate([{"role": "user", "content": "What's ur name?"}]))