import bittensor as bt
from prompting.utils.consts import VALIDATOR_MODEL_ID, TASKS, TASKS_P
from prompting.forward import forward
from prompting.llms import vLLMPipeline
from prompting.base.validator import BaseValidatorNeuron
from prompting.rewards import RewardPipeline
from prompting.tasks.translate import TranslationPipeline



class Validator(BaseValidatorNeuron):
    """
    Text prompt validator neuron.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.llm_pipeline = vLLMPipeline(
            model_id=VALIDATOR_MODEL_ID,
            gpus=self.config.neuron.gpus,
            llm_max_allowed_memory_in_gb=self.config.neuron.llm_max_allowed_memory_in_gb,
            device=self.device,
            mock=self.config.mock,
        )        
        self.translation_pipeline = TranslationPipeline()

        if abs(1-sum(TASKS_P)) > 0.001:
            raise ValueError("Task probabilities do not sum to 1.")

        # Filter out tasks with 0 probability
        self.active_tasks = [
            task
            for task, p in zip(TASKS, TASKS_P)
            if p > 0
        ]
        # Load the reward pipeline
        self.reward_pipeline = RewardPipeline(
            selected_tasks=self.active_tasks, device=self.device
        )

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)

    def __enter__(self):
        if self.config.no_background_thread:
            bt.logging.warning("Running validator in main thread.")
            self.run()
        else:
            self.run_in_background_thread()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")
