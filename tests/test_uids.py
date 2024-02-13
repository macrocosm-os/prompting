
import torch
import random
import pytest
from types import SimpleNamespace
from typing import List, Set
from prompting.utils.uids import get_random_uids

def make_mock_neuron(sample_size, unique_coldkey_prob=0, unique_ip_prob=0, add_validator=False, add_inactive=False, add_validator_above_limit=False):

    axons = [
        SimpleNamespace(coldkey="a", ip="0.0.0.1", is_serving=True),
        SimpleNamespace(coldkey="a", ip="0.0.0.2", is_serving=True),
        SimpleNamespace(coldkey="b", ip="0.0.0.1", is_serving=True),
        SimpleNamespace(coldkey="b", ip="0.0.0.2", is_serving=True),
        SimpleNamespace(coldkey="c", ip="1.0.0.0", is_serving=True),
        SimpleNamespace(coldkey="d", ip="0.1.0.0", is_serving=True),
    ]
    if add_validator:
        axons.append(SimpleNamespace(coldkey="e", ip="0.0.1.0", is_serving=True))
    if add_inactive:
        axons.append(SimpleNamespace(coldkey="f", ip="0.0.0.1", is_serving=False))
    if add_validator_above_limit:
        axons.append(SimpleNamespace(coldkey="g", ip="1.1.1.1", is_serving=True))

    metagraph = SimpleNamespace(
        axons = axons,
        coldkeys = [axon.coldkey for axon in axons],
        validator_permit = torch.ones(len(axons), dtype=torch.bool),
        S = torch.zeros(len(axons)),
        n = torch.tensor(len(axons))
    )
    
    if add_validator_above_limit:
        metagraph.S[-1] = 2000

    return SimpleNamespace(
        uid = 6,
        config = SimpleNamespace(
            neuron = SimpleNamespace(
                sample_size = sample_size,
                vpermit_tao_limit = 1000,
                unique_coldkey_prob = unique_coldkey_prob,
                unique_ip_prob = unique_ip_prob,
            )
        ),
        metagraph = metagraph
    )

ALL_IPS = {'0.0.0.1', '0.0.0.2', '1.0.0.0', '0.1.0.0'}
ALL_COLDKEYS = {'a', 'b', 'c', 'd'}

@pytest.mark.parametrize(
    "unique_coldkey_prob, unique_ip_prob, sample_size, expected_coldkeys, expected_ips, expected_count", [
        (0, 0, 10, ALL_COLDKEYS, ALL_IPS, 6),
        (0, 0, 8,  ALL_COLDKEYS, ALL_IPS, 6),
        (0, 1, 8,  None,         ALL_IPS, 4),
        (1, 0, 8,  ALL_COLDKEYS, None,    4),
        (1, 1, 8,  ALL_COLDKEYS, ALL_IPS, 4),
        ])
@pytest.mark.parametrize('add_validator', [True, False])
@pytest.mark.parametrize('add_inactive', [True, False])
@pytest.mark.parametrize('add_validator_above_limit', [True, False])
@pytest.mark.parametrize('trial', range(5))
def test_get_random_uids(unique_coldkey_prob: bool, unique_ip_prob: bool, sample_size: int, expected_coldkeys: Set[str], expected_ips: Set[str], expected_count: int, add_validator:bool, add_inactive: bool, add_validator_above_limit: bool, trial: int):

    mock_neuron = make_mock_neuron(sample_size=sample_size, unique_coldkey_prob=unique_coldkey_prob, unique_ip_prob=unique_ip_prob, add_validator=add_validator, add_inactive=add_inactive, add_validator_above_limit=add_validator_above_limit)
    uids = get_random_uids(mock_neuron).tolist()
    coldkeys = [mock_neuron.metagraph.coldkeys[uid] for uid in uids]
    ips = [mock_neuron.metagraph.axons[uid].ip for uid in uids]

    assert len(uids) == expected_count
    assert expected_coldkeys is None or set(coldkeys) == expected_coldkeys
    assert expected_ips is None or set(ips) == expected_ips

