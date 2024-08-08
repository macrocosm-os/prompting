from loguru import logger


class OpenAIUtils:
    def __init__(self) -> None:
        pass

    def get_cost_logging(self, cb):
        logger.info(f"Total Tokens: {cb.total_tokens}")
        logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
        logger.info(f"Completion Tokens: {cb.completion_tokens}")
        logger.info(f"Total Cost (USD): ${round(cb.total_cost,4)}")

        self.accumulated_total_tokens += cb.total_tokens
        self.accumulated_prompt_tokens += cb.prompt_tokens
        self.accumulated_completion_tokens += cb.completion_tokens
        self.accumulated_total_cost += cb.total_cost

        return {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost": cb.total_cost,
            "accumulated_total_tokens": self.accumulated_total_tokens,
            "accumulated_prompt_tokens": self.accumulated_prompt_tokens,
            "accumulated_completion_tokens": self.accumulated_completion_tokens,
            "accumulated_total_cost": self.accumulated_total_cost,
        }
