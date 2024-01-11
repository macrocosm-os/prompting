import torch
import bittensor as bt


def update_ema_scores(self, uids, final_rewards):
    # TODO: Implement this
    bt.logging.info("Updating EMA scores...")

    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
        0, uids, final_rewards
    ).to(self.device)

    # Update moving_averaged_scores with rewards produced by this step.
    # shape: [ metagraph.n ]
    alpha: float = self.config.neuron.moving_average_alpha
    self.moving_averaged_scores: torch.FloatTensor = alpha * scattered_rewards + (
        1 - alpha
    ) * self.moving_averaged_scores.to(self.device)
