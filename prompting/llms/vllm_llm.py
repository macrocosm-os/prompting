import random
import numpy as np
import torch
from vllm import LLM, SamplingParams
from loguru import logger


class ReproducibleVLLM:
    def __init__(
        self,
        model_id: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ",
        device: str = "cuda:0",
        sampling_params: dict[str, str | float | int | bool] | None = None,
    ):
        """Deterministic VLLM model."""
        self._device = device
        self.model_id = model_id
        self.sampling_params = {} if sampling_params else sampling_params

        # VLLM specific initialization
        # gpu_memory_utilization = 0.9  # Default high utilization since VLLM is memory efficient
        self.model = LLM(
            model=model_id,
            tensor_parallel_size=1,  # Single GPU by default
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.7,
            max_model_len=1000,
        )

        # Store tokenizer from VLLM for consistency
        self.tokenizer = self.model.get_tokenizer()

    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        sampling_params: dict[str, str | float | int | bool] | None = None,
        seed: int | None = None,
        continue_last_message: bool = False,
    ) -> str:
        """Generate text with optimized performance using VLLM."""
        self.set_random_seeds(seed)

        # Convert chat messages to prompt string using tokenizer's chat template
        if isinstance(messages, list) and isinstance(messages[0], dict):
            try:
                # Try using the tokenizer's chat template
                prompt = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=not continue_last_message,
                    continue_final_message=continue_last_message,
                )
            except (AttributeError, NotImplementedError) as e:
                # Fallback for tokenizers without chat template support
                logger.warning(f"Chat template not supported for model {self.model_id}, using default format")
                prompt = ""
                for msg in messages:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    if role == "system":
                        prompt += f"System: {content}\n"
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"
                prompt = prompt.strip()
        else:
            prompt = messages[0] if isinstance(messages, list) else messages

        # Convert sampling parameters to VLLM format
        params = sampling_params if sampling_params else self.sampling_params
        vllm_params = SamplingParams(
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 1.0),
            max_tokens=params.get("max_new_tokens", 100),
            presence_penalty=params.get("presence_penalty", 0.0),
            frequency_penalty=params.get("frequency_penalty", 0.0),
            top_k=params.get("top_k", -1),
            logprobs=params.get("logprobs", None),
        )

        # Generate using VLLM
        outputs = self.model.generate(prompt, vllm_params)

        if not outputs:
            return ""

        # Return just the generated text without the prompt
        result = outputs[0].outputs[0].text
        return result

    def generate_logits(
        self,
        messages: list[str] | list[dict[str, str]],
        top_n: int = 10,
        sampling_params: dict[str, str | float | int | bool] | None = None,
        seed: int | None = None,
        continue_last_message: bool = False,
    ) -> dict[str, float]:
        """
        Generate logits for the next token prediction.

        Args:
            messages: Input messages or text
            top_n: Number of top logits to return (default: 10)
            sampling_params: Generation parameters
            seed: Random seed for reproducibility
            continue_last_message: Whether to continue the last message in chat format

        Returns:
            dict: Dictionary mapping tokens to their log probabilities
        """
        self.set_random_seeds(seed)

        # Convert chat messages to prompt string using tokenizer's chat template
        if isinstance(messages, list) and isinstance(messages[0], dict):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=not continue_last_message,
                    continue_final_message=continue_last_message,
                )
            except (AttributeError, NotImplementedError) as e:
                logger.warning(f"Chat template not supported for model {self.model_id}, using default format")
                prompt = ""
                for msg in messages:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    if role == "system":
                        prompt += f"System: {content}\n"
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"
                prompt = prompt.strip()
        else:
            prompt = messages[0] if isinstance(messages, list) else messages

        # Set up sampling parameters for logit generation
        params = sampling_params if sampling_params else self.sampling_params
        vllm_params = SamplingParams(
            temperature=1.0,  # Use temperature 1.0 for raw logits
            top_p=1.0,  # No filtering
            max_tokens=1,  # We only need one token for logits
            top_k=50,
            logprobs=top_n,  # Get top_n logprobs
        )

        # Generate using VLLM
        outputs = self.model.generate(prompt, vllm_params)

        if not outputs or not outputs[0].outputs[0].logprobs:
            return {}

        # Extract logprobs from the first token
        logprobs = outputs[0].outputs[0].logprobs[0]

        logprobs_list = [(k, v.logprob) for k, v in logprobs.items()]
        sorted_logprobs = sorted(logprobs_list, key=lambda x: x[1], reverse=True)

        top_token_ids = [x[0] for x in sorted_logprobs]
        top_logprob_values = [x[1] for x in sorted_logprobs]

        step_logprobs = {
            "top_tokens": [self.tokenizer.decode([tid]) for tid in top_token_ids],
            "top_logprobs": top_logprob_values,
        }

        # Create dictionary of token to logprob mapping
        token_logprobs = {
            token: logprob for token, logprob in zip(step_logprobs["top_tokens"], step_logprobs["top_logprobs"])
        }

        return token_logprobs

    def set_random_seeds(self, seed: int | None = 42):
        """Set random seeds for reproducibility across all relevant libraries."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
