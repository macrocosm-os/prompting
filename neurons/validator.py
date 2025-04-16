import asyncio
import sys
from multiprocessing.managers import AcquirerProxy

import netaddr
import requests
import torch
import torch.multiprocessing as mp
import wandb
from bittensor.core.extrinsics.serving import serve_extrinsic
from loguru import logger

from prompting.llms.model_manager import AsyncModelScheduler, ModelManager
from prompting.rewards.scoring import task_scorer

# ruff: noqa: E402
from shared import settings
from shared.logging import init_wandb

settings.shared_settings = settings.SharedSettings.load(mode="validator")

from prompting.llms.utils import GPUInfo

logger.remove()
logger.add("logfile.log", rotation="100 MB", retention="10 days", level="DEBUG")
logger.add("err.log", rotation="100 MB", retention="10 days", level="WARNING")
logger.add(sys.stderr, level=settings.shared_settings.LOG_LEVEL)

torch.multiprocessing.set_start_method("spawn", force=True)


async def create_loop_process(
    model_scheduler: AsyncModelScheduler,
    task_queue: list,
    scoring_queue: list,
    reward_events: list,
    miners_dict: dict,
    mp_lock: AcquirerProxy,
) -> None:
    # Load settings and initialize external services.
    settings.shared_settings = settings.SharedSettings.load(mode="validator")
    if settings.shared_settings.WANDB_ON:
        init_wandb(neuron="validator")

    # A list to keep references to all the tasks we spawn, so they can be cancelled later.
    all_tasks: list[asyncio.Task] = []

    async def cleanup():
        logger.info("Cleaning up resources...")
        for t in all_tasks:
            t.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        if settings.shared_settings.WANDB_ON:
            wandb.finish()
            logger.info("WandB run finished.")

    async def spawn_loops(task_queue: list, scoring_queue: list, reward_events: list, miners_dict: dict):
        # Import modules that are local to this scope.
        from prompting.tasks.task_creation import task_loop
        from shared.profiling import profiler

        logger.info("Starting loops...")
        # Create tasks and give them descriptive names.
        profile: asyncio.Task = asyncio.create_task(profiler.print_stats(), name="Profiler")
        task_loop_task: asyncio.Task = asyncio.create_task(
            task_loop.start(task_queue, scoring_queue, miners_dict, simultaneous_loops=1), name="TaskLoop"
        )
        model_scheduler_task: asyncio.Task = asyncio.create_task(
            model_scheduler.start(scoring_queue), name="ModelScheduler"
        )
        task_scorer_task: asyncio.Task = asyncio.create_task(
            task_scorer.start(model_scheduler, scoring_queue, reward_events, mp_lock=mp_lock, simultaneous_loops=1),
            name="TaskScorer",
        )
        all_tasks.extend([profile, task_loop_task, model_scheduler_task, task_scorer_task])

        try:
            while True:
                await asyncio.sleep(10)
                logger.debug(
                    f"Task Queue {len(task_queue)}. Scoring Queue {len(scoring_queue)}. Reward Events {len(reward_events)}"
                )
                if model_scheduler.memory_error is not None:
                    raise model_scheduler.memory_error
        except asyncio.CancelledError:
            logger.info("spawn_loops received cancellation signal.")
            raise

    try:
        await spawn_loops(task_queue, scoring_queue, reward_events, miners_dict)
    except MemoryError as e:
        logger.error(f"MemoryError encountered. Terminating program: {e}")
        await cleanup()
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Terminating loop process: {e}")
    finally:
        await cleanup()


def start_api(
    scoring_queue: list,
    reward_events: list,
    miners_dict: dict,
):
    from prompting.api.api import start_scoring_api  # noqa: F401

    async def start():
        try:
            external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
            netaddr.IPAddress(external_ip)

            serve_success = serve_extrinsic(
                subtensor=settings.shared_settings.SUBTENSOR,
                wallet=settings.shared_settings.WALLET,
                ip=external_ip,
                port=settings.shared_settings.SCORING_API_PORT,
                protocol=4,
                netuid=settings.shared_settings.NETUID,
            )

            logger.debug(f"Serve success: {serve_success}")
        except Exception as e:
            logger.warning(f"Failed to serve scoring api to chain: {e}")
        await start_scoring_api(task_scorer, scoring_queue, reward_events, miners_dict)

        while True:
            await asyncio.sleep(10)

    asyncio.run(start())


def start_task_sending_loop(
    task_queue: list,
    scoring_queue: list,
    miners_dict: dict,
):
    async def spawn_loops(task_queue, scoring_queue, miners_dict: dict):
        from prompting.tasks.task_sending import TaskSender

        logger.info("Starting task sending loop in validator...")
        task_sender = TaskSender()
        asyncio.create_task(task_sender.start(task_queue, scoring_queue, miners_dict, simultaneous_loops=1))
        logger.debug("Task sending loop started")
        while True:
            await asyncio.sleep(5)
            logger.debug("Task sending loop is running")

    try:
        logger.info("Starting task sending loop in validator...")
        asyncio.run(spawn_loops(task_queue, scoring_queue, miners_dict))

    except Exception as e:
        logger.exception(f"Task sending loop error: {e}")
        raise


def start_availability_checking_loop(miners_dict: dict):
    async def spawn_loops(miners_dict: dict):
        from prompting.miner_availability.miner_availability import availability_checking_loop

        logger.info("Starting availability checking loop in validator...")
        asyncio.create_task(availability_checking_loop.start(miners_dict))
        while True:
            await asyncio.sleep(5)
            logger.debug("Availability checking loop is running")

    try:
        logger.info("Starting availability checking loop in validator...")
        asyncio.run(spawn_loops(miners_dict))

    except Exception as e:
        logger.exception(f"Availability checking loop error: {e}")
        raise


def start_weight_setter_loop(reward_events):
    async def spawn_loops(reward_events):
        from prompting.weight_setting.weight_setter import weight_setter

        logger.info("Starting weight setter loop in validator...")
        asyncio.create_task(weight_setter.start(reward_events))
        while True:
            await asyncio.sleep(5)
            logger.debug("Weight setter loop is running")

    try:
        logger.info("Starting weight setter loop in validator...")
        asyncio.run(spawn_loops(reward_events))

    except Exception as e:
        logger.exception(f"Weight setter loop error: {e}")
        raise


async def main(
    cache_rewards: list | None = None,
    cache_scores: list | None = None,
    cache_tasks: list | None = None,
    cache_miners: dict | None = None,
):
    # will start checking the availability of miners at regular intervals, needed for API and Validator
    with mp.Manager() as manager:
        reward_events = manager.list(list(cache_rewards) if cache_rewards else [])
        scoring_queue = manager.list(list(cache_scores) if cache_scores else [])
        task_queue = manager.list(list(cache_tasks) if cache_tasks else [])
        miners_dict = manager.dict(dict(cache_miners) if cache_miners else {})
        mp_lock = manager.Lock()
        processes: list[mp.Process] = []
        tasks: list[asyncio.Task] = []

        model_scheduler = AsyncModelScheduler(llm_model_manager=ModelManager(), mp_lock=mp_lock, sync=True)

        try:
            # Start checking the availability of miners at regular intervals
            if settings.shared_settings.DEPLOY_SCORING_API and not settings.shared_settings.NEURON_DISABLE_SET_WEIGHTS:
                # Use multiprocessing to bypass API blocking issue
                api_process = mp.Process(
                    target=start_api, args=(scoring_queue, reward_events, miners_dict), name="APIProcess"
                )
                api_process.start()
                processes.append(api_process)

            availability_process = mp.Process(
                target=start_availability_checking_loop,
                args=(miners_dict,),
                name="AvailabilityProcess",
            )
            availability_process.start()
            processes.append(availability_process)

            loop_task = asyncio.create_task(
                create_loop_process(
                    model_scheduler=model_scheduler,
                    task_queue=task_queue,
                    scoring_queue=scoring_queue,
                    reward_events=reward_events,
                    miners_dict=miners_dict,
                    mp_lock=mp_lock,
                )
            )
            tasks.append(loop_task)

            sending_task = mp.Process(
                target=start_task_sending_loop,
                args=(task_queue, scoring_queue, miners_dict),
                name="SendingTaskProcess",
            )
            sending_task.start()
            processes.append(sending_task)

            weight_setter_process = mp.Process(
                target=start_weight_setter_loop,
                args=(reward_events,),
                name="WeightSetterProcess",
            )
            weight_setter_process.start()
            processes.append(weight_setter_process)

            GPUInfo.log_gpu_info()

            step = 0
            while True:
                await asyncio.sleep(30)
                block = settings.shared_settings.SUBTENSOR.get_current_block()
                if (
                    block - settings.shared_settings.METAGRAPH.last_update[settings.shared_settings.UID] > 500
                    and step > 120
                ):
                    last_update_block = settings.shared_settings.METAGRAPH.last_update[settings.shared_settings.UID]
                    logger.warning(
                        f"Metagraph hasn't been updated for {block - last_update_block} blocks. "
                        f"Staled block: {block}, Last update: {last_update_block}"
                    )
                    break
                step += 1

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            raise
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks)

            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            sys.exit(1)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
