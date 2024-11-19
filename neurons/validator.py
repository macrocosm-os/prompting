# ruff: noqa: E402
import asyncio
import time
from prompting import settings
from prompting.utils.profiling import profiler

settings.settings = settings.Settings.load(mode="validator")
settings = settings.settings

from loguru import logger
from prompting.base.validator import BaseValidatorNeuron
from prompting.base.forward import log_stream_results, handle_response
from prompting.base.dendrite import DendriteResponseEvent, StreamPromptingSynapse
from prompting.tasks.task_creation import task_loop
from prompting.utils.logging import ValidatorLoggingEvent, ErrorLoggingEvent
from prompting.rewards.scoring import task_scorer
from prompting.miner_availability.miner_availability import availability_checking_loop, miner_availabilities
from prompting.llms.model_manager import model_scheduler
from prompting.utils.timer import Timer
from prompting.mutable_globals import scoring_queue
from prompting import mutable_globals
from prompting.tasks.base_task import BaseTextTask
from prompting.organic.organic_loop import start_organic
from prompting.weight_setting.weight_setter import weight_setter
from prompting.llms.utils import GPUInfo
from prompting.base.epistula import query_miners

NEURON_SAMPLE_SIZE = 100


def run_dendrite_and_handle_response_sync(uids, *args, **kwargs):
    async def run_dendrite_and_handle_response(uids, *args, **kwargs):
        # Run DENDRITE and handle_response sequentially within the main event loop
        streams_responses = await settings.DENDRITE(*args, **kwargs)
        # Handle the responses synchronously
        stream_results = await handle_response(stream_results_dict=dict(zip(uids, streams_responses)))
        return stream_results

    # Synchronously run the async function
    return asyncio.run(run_dendrite_and_handle_response(uids, *args, **kwargs))


class Validator(BaseValidatorNeuron):
    """Text prompt validator neuron."""

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.load_state()
        self._lock = asyncio.Lock()
        start_organic(self.axon)
        self.time_of_block_sync = None

    @property
    def estimate_block(self):
        """
        Estimate the current block number based on the time since the last block sync.

        Returns:
            Optional[int]: The estimated block number or None if an error occurs.
        """

        if self.time_of_block_sync is None:
            block = self.block
            return block

        # Calculate the block based on the time since the last block
        time_since_last_block = time.time() - self.time_of_block_sync
        # A block happens every 12 seconds
        blocks_since_last_block = time_since_last_block // 12
        estimated_block = int(self._block + blocks_since_last_block)

        return estimated_block

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
        while len(scoring_queue) > settings.SCORING_QUEUE_LENGTH_THRESHOLD:
            logger.debug("Scoring queue is full. Waiting 1 second...")
            await asyncio.sleep(1)
        while len(mutable_globals.task_queue) == 0:
            logger.warning("No tasks in queue. Waiting 1 second...")
            await asyncio.sleep(1)
        try:
            # get task from the task queue
            mutable_globals.task_queue: list[BaseTextTask]
            task = mutable_globals.task_queue.pop(0)

            # send the task to the miners and collect the responses
            with Timer() as timer:
                response_event = await self.collect_responses(task=task)
            if response_event is None:
                logger.warning("No response event collected. This should not be happening.")
                return
            logger.debug(f"Collected responses in {timer.elapsed_time:.2f} seconds")

            # scoring_manager will score the responses as and when the correct model is loaded
            task_scorer.add_to_queue(
                task=task,
                response=response_event,
                dataset_entry=task.dataset_entry,
                block=self.estimate_block,
                step=self.step,
                task_id=task.task_id,
            )

            # Log the step event.
            return ValidatorLoggingEvent(
                block=self.estimate_block,
                step=self.step,
                step_time=timer.elapsed_time,
                response_event=response_event,
                task_id=task.task_id,
            )

        except Exception as ex:
            logger.exception(ex)
            return ErrorLoggingEvent(
                error=str(ex),
            )

    async def collect_responses(self, task: BaseTextTask) -> DendriteResponseEvent | None:
        # Get the list of uids and their axons to query for this step.
        uids = miner_availabilities.get_available_miners(task=task, model=task.llm_model_id, k=NEURON_SAMPLE_SIZE)
        logger.debug(f"🔍 Querying uids: {uids}")
        if len(uids) == 0:
            logger.warning("No available miners. This should already have been caught earlier.")
            return


        body = {"seed": task.seed, "model": task.llm_model_id, "roles": ["user"], "messages": [task.query]}
        body_bytes = json.dumps(body).encode("utf-8")
        stream_results = query_miners(task.__class__.__name__, uids, body)

        log_stream_results(stream_results)


        response_event = DendriteResponseEvent(
            stream_results=stream_results, uids=uids, timeout=settings.NEURON_TIMEOUT
        )
        return response_event

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


async def main():
    GPUInfo.log_gpu_info()
    # start profiling
    asyncio.create_task(profiler.print_stats())

    # start rotating LLM models
    asyncio.create_task(model_scheduler.start())

    # start creating tasks
    asyncio.create_task(task_loop.start())

    # will start checking the availability of miners at regular intervals
    asyncio.create_task(availability_checking_loop.start())

    # sets weights at regular intervals (synchronised between all validators)
    asyncio.create_task(weight_setter.start())

    # start scoring tasks in separate loop
    asyncio.create_task(task_scorer.start())
    # TODO: Think about whether we want to store the task queue locally in case of a crash
    # TODO: Possibly run task scorer & model scheduler with a lock so I don't unload a model whilst it's generating
    # TODO: Make weight setting happen as specific intervals as we load/unload models
    with Validator() as v:
        while True:
            logger.info(
                f"Validator running:: network: {settings.SUBTENSOR.network} "
                f"| block: {v.estimate_block} "
                f"| step: {v.step} "
                f"| uid: {v.uid} "
                f"| last updated: {v.estimate_block - settings.METAGRAPH.last_update[v.uid]} "
                f"| vtrust: {settings.METAGRAPH.validator_trust[v.uid]:.3f} "
                f"| emission {settings.METAGRAPH.emission[v.uid]:.3f}"
            )
            time.sleep(5)

            if v.should_exit:
                logger.warning("Ending validator...")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
    # will start rotating the different LLMs in/out of memory
