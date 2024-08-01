import time
import bittensor as bt
from prompting.validator import Validator


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as v:
        while True:
            bt.logging.info(
                f"Validator running:: network: {v.subtensor.network} | block: {v.block} | step: {v.step} | uid: {v.uid} | last updated: {v.block-v.metagraph.last_update[v.uid]} | vtrust: {v.metagraph.validator_trust[v.uid]:.3f} | emission {v.metagraph.emission[v.uid]:.3f}"
            )
            time.sleep(5)

            if v.should_exit:
                bt.logging.warning("Ending validator...")
                break
