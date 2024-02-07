import torch
import numpy as np
import bittensor as bt
from typing import List


def get_random_uids(
    self, exclude: List[int] = None
) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If self.config.neuron.sample_size is larger than the number of available `uids`, the function will return all available `uids`.
    """

    uids = []
    coldkeys = {}
    ips = {}
    # shuffled list of all UIDs
    all_uids = np.random.choice(range(self.metagraph.n.item()), size=self.metagraph.n.item(), replace=False)
    all_coldkeys = self.metagraph.coldkeys
    all_ips = [axon.ip for axon in self.metagraph.axons]
    for uid in all_uids:

        if exclude is not None and uid in exclude:
            continue

        # Filter non serving axons.
        if not self.metagraph.axons[uid].is_serving:
            bt.logging.debug(f"uid: {uid} is not serving")
            continue

        # Filter validator permit > 1024 stake.
        if self.metagraph.validator_permit[uid] and self.metagraph.S[uid] > self.config.neuron.vpermit_tao_limit:
            bt.logging.debug(f"uid: {uid} has vpermit and stake ({self.metagraph.S[uid]}) > {self.config.neuron.vpermit_tao_limit}")
            continue

        # get the coldkey for the uid
        coldkey = all_coldkeys[uid]
        ip = all_ips[uid]
        # get the number of times the coldkey has been queried in the current step
        ck_counts = coldkeys.get(coldkey,0)
        # get the number of times the ip has been queried in the current step
        ip_counts = ips.get(ip,0)
        # if it's already been queried query again with some smaller probability
        if ck_counts > 0 or ip_counts > 0:

            # here we use the probability of not querying the same coldkey
            # for example if unique_coldkey_prob = 0.9 and the coldkey has already been queried 2 times in this forward pass, then the probability of querying the same coldkey again is (1-0.9)^2=0.01
            ck_threshold = (1-self.config.neuron.unique_coldkey_prob) ** ck_counts
            ip_threshold = (1-self.config.neuron.unique_ip_prob) ** ip_counts

            # Take the product of the two probabilities as the likelihood of querying the same coldkey and ip again
            if np.random.random() > ck_threshold * ip_threshold:
                continue

        coldkeys[coldkey] = coldkeys.get(coldkey, 0 ) + 1
        ips[ip] = ips.get(ip, 0 ) + 1

        uids.append(uid)
        if len(uids) == self.config.neuron.sample_size:
            break

    self._selected_coldkeys = coldkeys
    self._selected_ips = ips
    if len(uids) < self.config.neuron.sample_size:
        bt.logging.warning(f"Only {len(uids)} uids available for querying, requested {k}.")
    return torch.tensor(uids)
