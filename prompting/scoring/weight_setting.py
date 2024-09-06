from prompting.mutable_globals import reward_events
from prompting.rewards.reward import BaseRewardConfig
from loguru import logger
from prompting.base.loop_runner import AsyncLoopRunner


class WeightSetter(AsyncLoopRunner):
    # sync = True makes it such that validators set weights at the same time exactly
    sync: bool = True

    async def run_step(self):
        logger.debug(f"Weight setting rewards for length: {len(reward_events)}")
        for task_reward_events in reward_events:
            rewards = BaseRewardConfig.final_rewards(task_reward_events)
            logger.debug(f"Weight setting rewards: {rewards}")


weight_setter = WeightSetter()
