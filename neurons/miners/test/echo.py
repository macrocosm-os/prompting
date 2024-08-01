import time
from prompting.miners import EchoMiner


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with EchoMiner() as miner:
        while True:
            miner.log_status()
            time.sleep(5)
