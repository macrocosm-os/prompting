import time
from prompting.miners import MockMiner


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with MockMiner() as miner:
        while True:
            miner.log_status()
            time.sleep(5)
