# ruff: noqa: E402
import asyncio
import time

from prompting import settings

settings.settings = settings.Settings(mode="validator")
settings = settings.settings
from loguru import logger
from prompting.base.validator import BaseValidatorNeuron
from neurons.forward import log_stream_results, handle_response
from prompting.base.dendrite import DendriteResponseEvent, StreamPromptingSynapse
from prompting.tasks.task_registry import TaskRegistry
from prompting.utils.logging import log_event
from prompting.utils.logging import ValidatorLoggingEvent, ErrorLoggingEvent
from prompting.rewards.scoring import task_scorer
from prompting.miner_availability.miner_availability import checking_loop, miner_availabilities
from prompting.llms.model_manager import model_scheduler
from prompting.utils.timer import Timer

NEURON_SAMPLE_SIZE = 100
SCORING_QUEUE_LENGTH_THRESHOLD = 10

# will start rotating the different LLMs in/out of memory
asyncio.run(model_scheduler.start())

# will start checking the availability of miners at regular intervals
asyncio.run(checking_loop.start())

# start scoring tasks in separate loop
asyncio.run(task_scorer.start())


class Validator(BaseValidatorNeuron):
    """
    Text prompt validator neuron.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.load_state()
        self._lock = asyncio.Lock()

    async def run_step(self, k: int, timeout: float) -> ValidatorLoggingEvent | ErrorLoggingEvent | None:
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

        if len(task_scorer.scoring_queue) > SCORING_QUEUE_LENGTH_THRESHOLD:
            logger.debug("Scoring queue is full. Skipping task generation.")
            return None

        try:
            # Getting task & Dataset
            while True:
                try:
                    task, dataset = TaskRegistry.create_random_task_with_dataset()
                    break
                except Exception as ex:
                    logger.exception(ex)

            if len(miner_availabilities.get_available_miners(task=task, model=task.model_id)) == 0:
                logger.debug(
                    f"No available miners for Task: {task.__class__.__name__} and Model ID: {task.model_id}. Skipping step."
                )
                return None

            if not (dataset_entry := dataset.random()):
                logger.warning(f"Dataset {dataset.__class__.__name__} returned None. Skipping step.")
                return None

            # Generate the query and reference for the task
            if not task.query:
                logger.debug(f"Generating query for task: {task.__class__.__name__}.")
                query = task.make_query(dataset_entry=dataset_entry)

            # Record event start time.
            start_time = time.time()

            # Get the list of uids and their axons to query for this step.
            uids = miner_availabilities.get_available_miners(task=task, model=task.model_id, k=k)
            logger.debug(f"🔍 Querying uids: {uids}")
            if len(uids) == 0:
                logger.debug("No available miners. Skipping step.")
                return

            axons = [settings.METAGRAPH.axons[uid] for uid in uids]

            # Directly call dendrite and process responses in parallel
            streams_responses = await settings.DENDRITE(
                axons=axons,
                synapse=StreamPromptingSynapse(
                    task_name=task.__class__.__name__,
                    seed=task.seed,
                    target_model=task.model_id,
                    roles=["user"],
                    messages=[query],
                ),
                timeout=timeout,
                deserialize=False,
                streaming=True,
            )

            # Prepare the task for handling stream responses
            stream_results = await handle_response(stream_results_dict=dict(zip(uids, streams_responses)))

            log_stream_results(stream_results)

            # Encapsulate the responses in a response event (dataclass)
            response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=timeout)

            # scoring_manager will score the responses as and when the correct model is loaded
            task_scorer.add_to_queue(task=task, response=response_event, dataset_entry=dataset_entry)

            # Log the step event.
            return ValidatorLoggingEvent(
                block=self.block,
                step=self.step,
                step_time=time.time() - start_time,
                response_event=response_event,
                task_id=task.task_id,
            )

        except Exception as ex:
            logger.exception(ex)
            return ErrorLoggingEvent(
                error=str(ex),
            )

    async def forward(self):
        """
        Encapsulates a full conversation between the validator and miners. Contains one or more rounds of request-response.

        """
        logger.info("🚀 Starting forward loop...")
        with Timer() as timer:
            # in run_step, a task is generated and sent to the miners
            async with self._lock:
                event = await self.run_step(
                    k=NEURON_SAMPLE_SIZE,
                    timeout=settings.NEURON_TIMEOUT,
                )

        if not event:
            return

        event.forward_time = timer.elapsed_time
        log_event(event)

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

###


# ProgrammingTask -> return code
# Multiple choice -> return answer
# Online lookup -> return context from website
# Inference -> just run a model
# AgentTask -> uses the other model in an agentic to respond
