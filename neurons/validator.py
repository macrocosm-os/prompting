# ruff: noqa: E402
import asyncio
import time

from loguru import logger

from prompting import settings
from shared.profiling import profiler

settings.settings = settings.Settings.load(mode="validator")
settings = settings.settings

from prompting.api.api import start_api
from prompting.llms.model_manager import model_scheduler
from prompting.llms.utils import GPUInfo
from prompting.miner_availability.miner_availability import availability_checking_loop
from prompting.rewards.scoring import task_scorer
from prompting.tasks.task_creation import task_loop
from prompting.tasks.task_sending import task_sender
from prompting.weight_setting.weight_setter import weight_setter
import requests

NEURON_SAMPLE_SIZE = 100


async def main():
    # will start checking the availability of miners at regular intervals, needed for API and Validator
    asyncio.create_task(availability_checking_loop.start())

    if settings.DEPLOY_API:
        asyncio.create_task(start_api())

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
    # # TODO: Think about whether we want to store the task queue locally in case of a crash
    # # TODO: Possibly run task scorer & model scheduler with a lock so I don't unload a model whilst it's generating
    # # TODO: Make weight setting happen as specific intervals as we load/unload models
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
            print(v.block)
            time.sleep(5)

            if v.should_exit:
                logger.warning("Ending validator...")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
    # will start rotating the different LLMs in/out of memory
