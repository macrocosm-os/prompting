
import torch
import pytest
from types import SimpleNamespace
from prompting.utils.uids import get_random_uids


def make_mock_neuron(unique_coldkeys=False, unique_ips=False, vpermit_tao_limit=1000):

    axons = [
        SimpleNamespace(coldkey="a", ip="0.0.0.1", is_serving=True),
        SimpleNamespace(coldkey="a", ip="0.0.0.0", is_serving=True),
        SimpleNamespace(coldkey="b", ip="0.0.0.1", is_serving=True),
        SimpleNamespace(coldkey="b", ip="0.0.0.0", is_serving=True),
        SimpleNamespace(coldkey="c", ip="0.0.0.2", is_serving=True),
    ]
    metagraph = SimpleNamespace(
        axons = axons,
        validator_permit = torch.ones(len(axons), dtype=torch.bool),
        S = torch.zeros(len(axons)),
        n = torch.tensor(len(axons))
    )

    return SimpleNamespace(
        uid = 4,
        config = SimpleNamespace(
            neuron = SimpleNamespace(
                vpermit_tao_limit = vpermit_tao_limit,
                query_unique_coldkeys = unique_coldkeys,
                query_unique_ips = unique_ips,
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
def test_get_random_uids(unique_coldkeys, unique_ips, k, expected_result):

    mock_neuron = make_mock_neuron(unique_coldkeys, unique_ips)

    assert sorted(get_random_uids(mock_neuron, k).tolist()) == expected_result, "Incorrect uids returned."

