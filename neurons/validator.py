# ruff: noqa: E402
import asyncio
import time
from typing import Optional

import numpy as np

from prompting import settings

settings.settings = settings.Settings(mode="validator")
settings = settings.settings

import huggingface_hub
from loguru import logger

from neurons.forward import handle_response, log_stream_results
from prompting.base.dendrite import DendriteResponseEvent, StreamPromptingSynapse
from prompting.base.validator import BaseValidatorNeuron
from prompting.datasets.base import BaseDataset
from prompting.llms.vllm_llm import vLLMPipeline
from prompting.tasks.base_task import BaseTask
from prompting.tasks.task_registry import TaskRegistry
from prompting.utils.logging import ErrorEvent, ValidatorEvent, log_event
from prompting.utils.uids import get_random_uids

try:
    from organic_scoring.synth_dataset import SynthDatasetConversation

    from prompting.organic.organic_scoring_prompting import OrganicScoringPrompting
except ImportError:
    raise ImportError(
        "Could not import organic-scoring library.  Please install via poetry: `poetry install --extras 'validator'`"
    )

NEURON_SAMPLE_SIZE = 100


class Validator(BaseValidatorNeuron):
    """
    Text prompt validator neuron.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        logger.info("load_state()")
        self.load_state()
        self._lock = asyncio.Lock()

        self.llm_pipeline = vLLMPipeline(
            llm_model_id=settings.NEURON_MODEL_ID_VALIDATOR,
            llm_max_allowed_memory_in_gb=settings.NEURON_LLM_MAX_ALLOWED_MEMORY_IN_GB,
            llm_max_model_len=settings.LLM_MAX_MODEL_LEN,
            gpus=settings.NEURON_GPUS,
            device=self.device,
            mock=settings.MOCK,
        )

        if self.axon is None or settings.ORGANIC_DISABLED:
            logger.warning(
                "Organic scoring is not enabled. To enable, remove '--neuron.axon_off' and '--neuron.organic_disabled'"
            )
            return

        huggingface_hub.login(settings.HF_TOKEN)
        dataset = SynthDatasetConversation()
        if dataset.exception is not None:
            logger.error(
                "Organic scoring on synthetic data is disabled. Failed to load HF dataset.\nMake sure to:\n"
                "1. Accept License on: https://huggingface.co/datasets/lmsys/lmsys-chat-1m\n"
                "2. Create HF Access Token: https://huggingface.co/settings/tokens\n"
                "3. Set Access Token 'HF_TOKEN' in .env.validator\n"
            )
            dataset = None

        self._organic_scoring = OrganicScoringPrompting(
            axon=self.axon,
            synth_dataset=dataset,
            llm_pipeline=self.llm_pipeline,
            tokenizer=self.llm_pipeline.tokenizer,
            update_scores_fn=self.update_scores,
            get_random_uids_fn=lambda: get_random_uids(self, k=settings.ORGANIC_SAMPLE_SIZE, exclude=[]),
            get_step_fn=lambda: self.step,
            get_block_fn=lambda: self.block,
        )
        if self._organic_scoring is not None:
            self.loop.create_task(self._organic_scoring.start_loop())

    async def run_step(
        self, task: BaseTask, dataset: BaseDataset, k: int, timeout: float, exclude: Optional[list] = None
    ) -> ValidatorEvent | ErrorEvent | None:
        """Executes a single step of the agent, which consists of:
            - Getting a list of uids to query
            - Querying the network
            - Rewarding the network
            - Updating the scores
            - Logging the event
        Args:
            agent (HumanAgent): The agent to run the step for.
            roles (List[str]): The roles for the synapse.
            messages (List[str]): The messages for the synapse.
            k (int): The number of uids to query.
            timeout (float): The timeout for the queries.
            exclude (list, optional): The list of uids to exclude from the query. Defaults to [].
        """
        try:
            logger.debug("run_step", task.__class__.__name__)
            if not (dataset_entry := dataset.random()):
                logger.warning(f"Dataset {dataset.__class__.__name__} returned None. Skipping step.")
                return None
            # Generate the query and reference for the task
            query, reference = task.generate_query_reference(self.llm_pipeline, dataset_entry)
            # task.generate_reference(self.llm_pipeline)

            # Record event start time.
            start_time = time.time()

            # Get the list of uids to query for this step.
            uids = get_random_uids(self, k=k, exclude=exclude or [])

            axons = [settings.METAGRAPH.axons[uid] for uid in uids]

            # Directly call dendrite and process responses in parallel
            streams_responses = await self.dendrite(
                axons=axons,
                synapse=StreamPromptingSynapse(roles=["user"], messages=[query]),
                timeout=timeout,
                deserialize=False,
                streaming=True,
            )

            # Prepare the task for handling stream responses
            stream_results = await handle_response(
                stream_results_dict=dict(zip(uids, streams_responses)), tokenizer=self.llm_pipeline.tokenizer
            )

            log_stream_results(stream_results)

            # Encapsulate the responses in a response event (dataclass)
            response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=timeout)

            logger.info(f"Created DendriteResponseEvent:\n {response_event}")

            # Reward the responses and get the reward result (dataclass)
            # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
            reward_pipeline = TaskRegistry.get_task_reward(task)
            reward_events, penalty_events, rewards = reward_pipeline.apply(
                response_event=response_event, reference=reference, challenge=query
            )

            logger.info(f"Created RewardResult:\n {rewards}")

            best_response = response_event.completions[np.argmax(rewards)]

            self.update_scores(rewards, uids)

            # Log the step event.
            return ValidatorEvent(
                best=best_response or "",
                block=self.block,
                step=self.step,
                step_time=time.time() - start_time,
                reward_events=reward_events or [],
                penalty_events=penalty_events or [],
                reference=reference,
                challenge=query,
                task=task.name,
                rewards=rewards,
                response_event=response_event,
            )
        except Exception as ex:
            logger.exception(ex)
            return ErrorEvent(
                error=str(ex),
            )

    async def forward(self):
        """
        Encapsulates a full conversation between the validator and miners. Contains one or more rounds of request-response.

        """
        logger.info("🚀 Starting forward loop...")
        forward_start_time = time.time()

        while True:
            logger.info(f"📋 Selecting task... from {TaskRegistry.task_configs}")
            task_config = TaskRegistry.random()
            logger.info(f"📋 Creating {task_config.task.__name__} task... ")
            try:
                task, dataset = TaskRegistry.create_random_task_with_dataset()
                break
            except Exception as ex:
                logger.exception(ex)

        exclude_uids = []

        # when run_step is called, the agent updates its progress
        async with self._lock:
            event = await self.run_step(
                task=task,
                dataset=dataset,
                k=NEURON_SAMPLE_SIZE,
                timeout=settings.NEURON_TIMEOUT,
                exclude=exclude_uids,
            )

        # Adds forward time to event and logs it to wandb
        if not event:
            return

        event.forward_time = time.time() - forward_start_time
        log_event(event)

        # accepted_answer = event["best"] if random.random() < 0.5 else agent.task.reference

    def __enter__(self):
        if settings.NO_BACKGROUND_THREAD:
            logger.warning("Running validator in main thread.")
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
            logger.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as v:
        while True:
            logger.info(
                f"Validator running:: network: {settings.SUBTENSOR.network} "
                f"| block: {v.block} "
                f"| step: {v.step} "
                f"| uid: {v.uid} "
                f"| last updated: {v.block - settings.METAGRAPH.last_update[v.uid]} "
                f"| vtrust: {settings.METAGRAPH.validator_trust[v.uid]:.3f} "
                f"| emission {settings.METAGRAPH.emission[v.uid]:.3f}"
            )
            time.sleep(5)

            if v.should_exit:
                logger.warning("Ending validator...")
                break
