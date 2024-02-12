
import torch
import random
import pytest
from types import SimpleNamespace
from typing import List
from prompting.utils.uids import get_random_uids

def make_mock_neuron(sample_size, unique_coldkey_prob=False, unique_ip_prob=False, vpermit_tao_limit=1000):

    axons = [
        SimpleNamespace(coldkey="a", ip="0.0.0.1", is_serving=True),
        SimpleNamespace(coldkey="a", ip="0.0.0.0", is_serving=True),
        SimpleNamespace(coldkey="b", ip="0.0.0.1", is_serving=True),
        SimpleNamespace(coldkey="b", ip="0.0.0.0", is_serving=True),
        SimpleNamespace(coldkey="c", ip="0.0.0.2", is_serving=True), # validator
    ]
    metagraph = SimpleNamespace(
        axons = axons,
        coldkeys = [axon.coldkey for axon in axons],
        validator_permit = torch.ones(len(axons), dtype=torch.bool),
        S = torch.zeros(len(axons)),
        n = torch.tensor(len(axons))
    )

    return SimpleNamespace(
        uid = 4,
        config = SimpleNamespace(
            neuron = SimpleNamespace(
                sample_size = sample_size,
                vpermit_tao_limit = vpermit_tao_limit,
                unique_coldkey_prob = unique_coldkey_prob,
                unique_ip_prob = unique_ip_prob,
            )
        ),
        metagraph = metagraph
    )

@pytest.mark.parametrize(
    "unique_coldkeys, unique_ips, k, expected_result", [
        (False, False, 4, [0, 1, 2, 3]),
        (True, False, 2, [0, 2]),
        (False, True, 2, [0, 1]),
        (True, True, 2, [0, 3])
        ])
def test_get_random_uids(unique_coldkeys, unique_ips, sample_size, expected_result):

    mock_neuron = make_mock_neuron(sample_size, unique_coldkeys, unique_ips)

    assert sorted(get_random_uids(mock_neuron).tolist()) == expected_result, "Incorrect uids returned."

