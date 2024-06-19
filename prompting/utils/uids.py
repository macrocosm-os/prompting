import torch
import random
import bittensor as bt
from typing import List, Union


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
    coldkeys: set = None,
    ips: set = None,
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
        coldkeys (set): Set of coldkeys to exclude
        ips (set): Set of ips to exclude
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        bt.logging.debug(f"uid: {uid} is not serving")
        return False

    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid] and metagraph.S[uid] > vpermit_tao_limit:
        bt.logging.debug(
            f"uid: {uid} has vpermit and stake ({metagraph.S[uid]}) > {vpermit_tao_limit}"
        )
        return False

    if coldkeys and metagraph.axons[uid].coldkey in coldkeys:
        return False

    if ips and metagraph.axons[uid].ip in ips:
        return False

    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    coldkeys = set()
    ips = set()
    for uid in range(self.metagraph.n.item()):
        if uid == self.uid:
            continue

        uid_is_available = check_uid_availability(
            self.metagraph,
            uid,
            self.config.neuron.vpermit_tao_limit,
            coldkeys,
            ips,
        )
        if not uid_is_available:
            continue

        if self.config.neuron.query_unique_coldkeys:
            coldkeys.add(self.metagraph.axons[uid].coldkey)

        if self.config.neuron.query_unique_ips:
            ips.add(self.metagraph.axons[uid].ip)

        if exclude is None or uid not in exclude:
            candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    if 0 < len(candidate_uids) < k:
        bt.logging.warning(
            f"Requested {k} uids but only {len(candidate_uids)} were available. To disable this warning reduce the sample size (--neuron.sample_size)"
        )
        return torch.tensor(candidate_uids)
    elif len(candidate_uids) >= k:
        return torch.tensor(random.sample(candidate_uids, k))
    else:
        raise ValueError(f"No eligible uids were found. Cannot return {k} uids")


def get_top_incentive_uids(self, k: int, vpermit_tao_limit: int) -> List[int]:
    metagraph = self.validator.metagraph
    miners_uids = list(map(int, filter(lambda uid: check_uid_availability(metagraph, uid, vpermit_tao_limit),
        metagraph.uids)))
    
    # Builds a dictionary of uids and their corresponding incentives.
    all_miners_incentives = {
        "miners_uids": miners_uids,
        "incentives": list(map(lambda uid: metagraph.I[uid], miners_uids))
    }
    
    # Zip the uids and their corresponding incentives into a list of tuples.
    uid_incentive_pairs = list(zip(all_miners_incentives["miners_uids"], all_miners_incentives["incentives"]))

    # Sort the list of tuples by the incentive value in descending order.
    uid_incentive_pairs_sorted = sorted(uid_incentive_pairs, key=lambda x: x[1], reverse=True)

    # Extract the top uids.
    top_k_uids = [uid for uid, incentive in uid_incentive_pairs_sorted[:k]]
    
    return top_k_uids


def get_uids(self, sampling_mode: str, k: int, exclude: List[int] = []) -> Union[list[int], torch.LongTensor]:
    if sampling_mode == "random":
        uids = get_random_uids(self.validator, k=k, exclude=exclude or []).tolist()
        return uids
    if sampling_mode == "top_incentive":
        vpermit_tao_limit = self.validator.config.neuron.vpermit_tao_limit
        top_uids = get_top_incentive_uids(self, k=k, vpermit_tao_limit=vpermit_tao_limit)
        return top_uids
