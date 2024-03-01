# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import wandb
import time
import typing
import bittensor as bt
# Bittensor Miner Template:
import prompting
from prompting.protocol import PromptingSynapse
# import base miner class which takes care of most of the boilerplate
from prompting.base.miner import BaseMinerNeuron
from datetime import datetime

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)                
        self.identity_tags = None
         

    async def blacklist(
        self, synapse: PromptingSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (PromptingSynapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: PromptingSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (PromptingSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority
    
    def init_wandb(self):
        bt.logging.info("Initializing wandb...")
        
        uid = f"uid_{self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)}"
        net_uid = f"netuid_{self.config.netuid}"
        tags = [
            self.wallet.hotkey.ss58_address, 
            net_uid, 
            f"uid_{uid}",
            prompting.__version__,
            str(prompting.__spec_version__),
        ]
        
        run_name = None
        if self.identity_tags:
            # Add identity tags to run tags
            tags += self.identity_tags     

            # Create run name from identity tags       
            run_name_tags = [str(tag) for tag in self.identity_tags]            
            
            # Add uid, netuid and timestamp to run name
            run_name_tags += [uid, net_uid, datetime.now().strftime('%Y_%m_%d_%H_%M_%S')]

            # Compose run name
            run_name = '_'.join(run_name_tags)                
                    
        # inits wandb in case it hasn't been inited yet
        self.wandb_run = wandb.init(
            name=run_name,
            project=self.config.wandb.project_name,
            entity=self.config.wandb.entity,
            config=self.config,
            mode="online" if self.config.wandb.on else "offline",
            tags=tags,                
        )
    
    def log_event(self, timing: float, prompt: str, completion: str, system_prompt: str, extra_info: dict = {}):        
        if not getattr(self, "wandb_run", None):
            self.init_wandb()
        
        step_log = {
            "epoch_time": timing,
            # "block": self.last_epoch_block,
            "prompt": prompt,
            "completion": completion,
            "system_prompt": system_prompt,
            "uid": self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address),
            "stake": self.metagraph.S[self.uid].item(),
            "trust": self.metagraph.T[self.uid].item(),
            "incentive": self.metagraph.I[self.uid].item(),
            "consensus": self.metagraph.C[self.uid].item(),
            "dividends": self.metagraph.D[self.uid].item(),
            **extra_info
        }

        bt.logging.info('Logging event to wandb...', step_log)
        wandb.log(step_log)

    def log_status(self):
        m = self.metagraph
        bt.logging.info(f"Miner running:: network: {self.subtensor.network} | step: {self.step} | uid: {self.uid} | trust: {m.trust[self.uid]:.3f} | emission {m.emission[self.uid]:.3f}")

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            miner.log_status()
            time.sleep(5)

            if miner.should_exit:
                bt.logging.warning("Ending miner...")
                break
