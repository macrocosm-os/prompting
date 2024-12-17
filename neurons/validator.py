# ruff: noqa: E402
import asyncio
import multiprocessing as mp
import time

from loguru import logger

from prompting import settings
from shared.profiling import profiler

settings.settings = settings.Settings.load(mode="validator")
settings = settings.settings

from prompting import mutable_globals
from prompting.llms.model_manager import model_scheduler
from prompting.llms.utils import GPUInfo
from prompting.miner_availability.miner_availability import availability_checking_loop
from prompting.rewards.scoring import task_scorer
from prompting.tasks.task_creation import task_loop
from prompting.tasks.task_sending import task_sender
from prompting.weight_setting.weight_setter import weight_setter

NEURON_SAMPLE_SIZE = 100


async def start_dummy_endpoint():
    logger.debug("Starting Dummy Endpoint")
    while True:
        asyncio.sleep(2)


async def main():
    if settings.DEPLOY_API:
        # Use multiprocessing to bypass API blocking issue.
        api_process = mp.Process(target=lambda: asyncio.run(start_dummy_endpoint()))
        api_process.start()

    # will start checking the availability of miners at regular intervals, needed for API and Validator
    asyncio.create_task(availability_checking_loop.start())

    GPUInfo.log_gpu_info()
    # start profiling
    asyncio.create_task(profiler.print_stats())

    # start rotating LLM models
    asyncio.create_task(model_scheduler.start())

    # start creating tasks
    asyncio.create_task(task_loop.start())

    # will start checking the availability of miners at regular intervals
    asyncio.create_task(availability_checking_loop.start())

    # start sending tasks to miners
    asyncio.create_task(task_sender.start())

    # sets weights at regular intervals (synchronised between all validators)
    asyncio.create_task(weight_setter.start())

    # start scoring tasks in separate loop
    asyncio.create_task(task_scorer.start())
    # # TODO: Think about whether we want to store the task queue locally in case of a crash
    # # TODO: Possibly run task scorer & model scheduler with a lock so I don't unload a model whilst it's generating
    # # TODO: Make weight setting happen as specific intervals as we load/unload models
    start = time.time()
    await asyncio.sleep(60)
    while True:
        await asyncio.sleep(5)
        time_diff = -start + (start := time.time())
        logger.debug(f"Running {time_diff:.2f} seconds")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    if settings.DEPLOY_API:
        # Replace scoring queue with memory-shared list between processes, before starting the loops.
        manager = mp.Manager()
        mutable_globals.scoring_queue = manager.list()
    asyncio.run(main())
    # will start rotating the different LLMs in/out of memory
