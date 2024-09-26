# ruff: noqa: E402
import asyncio
import time
from prompting import settings

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

NEURON_SAMPLE_SIZE = 100


class Validator(BaseValidatorNeuron):
    """Text prompt validator neuron."""

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.load_state()
        self._lock = asyncio.Lock()
        start_organic(self.axon)

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
        if len(scoring_queue) > settings.SCORING_QUEUE_LENGTH_THRESHOLD:
            logger.debug("Scoring queue is full. Skipping task generation.")
            return None
        if len(mutable_globals.task_queue) == 0:
            logger.warning("No tasks in queue, skipping sending...")
            return
        try:
            # get task from the task queue
            mutable_globals.task_queue: list[BaseTextTask]
            task = mutable_globals.task_queue.pop(0)

            # send the task to the miners and collect the responses
            with Timer() as timer:
                response_event = await self.collect_responses(task=task)
            logger.debug(f"Collected responses in {timer.elapsed_time:.2f} seconds")

            # scoring_manager will score the responses as and when the correct model is loaded
            task_scorer.add_to_queue(
                task=task,
                response=response_event,
                dataset_entry=task.dataset_entry,
                block=self.block,
                step=self.step,
                task_id=task.task_id,
            )

            for uids, rewards in mutable_globals.rewards_and_uids:
                self.update_scores(uids=uids, rewards=rewards)
            mutable_globals.rewards_and_uids = []

            # Log the step event.
            return ValidatorLoggingEvent(
                block=self.block,
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

    async def collect_responses(self, task: BaseTextTask) -> DendriteResponseEvent:
        # Get the list of uids and their axons to query for this step.
        uids = miner_availabilities.get_available_miners(task=task, model=task.llm_model_id, k=NEURON_SAMPLE_SIZE)
        logger.debug(f"🔍 Querying uids: {uids}")
        if len(uids) == 0:
            logger.debug("No available miners. Skipping step.")
            return
        axons = [settings.METAGRAPH.axons[uid] for uid in uids]

        # Directly call dendrite and process responses in parallel
        synapse = StreamPromptingSynapse(
            task_name=task.__class__.__name__,
            seed=task.seed,
            target_model=task.llm_model_id,
            roles=["user"],
            messages=[task.query],
        )
        streams_responses = await settings.DENDRITE(
            axons=axons,
            synapse=synapse,
            timeout=settings.NEURON_TIMEOUT,
            deserialize=False,
            streaming=True,
        )

        # Prepare the task for handling stream responses
        stream_results = await handle_response(stream_results_dict=dict(zip(uids, streams_responses)))
        logger.debug(
            f"Non-empty responses: {len([r.completion for r in stream_results if len(r.completion) > 0])}\n"
            f"Empty responses: {len([r.completion for r in stream_results if len(r.completion) == 0])}"
        )

        log_stream_results(stream_results)

        # Encapsulate the responses in a response event (dataclass)
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
    # start rotating LLM models
    asyncio.create_task(model_scheduler.start())

    # start creating tasks
    asyncio.create_task(task_loop.start())

    # will start checking the availability of miners at regular intervals
    asyncio.create_task(availability_checking_loop.start())

    # start scoring tasks in separate loop
    asyncio.create_task(task_scorer.start())
    # TODO: Think about whether we want to store the task queue locally in case of a crash
    # TODO: Possibly run task scorer & model scheduler with a lock so I don't unload a model whilst it's generating
    # TODO: Make weight setting happen as specific intervals as we load/unload models
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


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
    # will start rotating the different LLMs in/out of memory
