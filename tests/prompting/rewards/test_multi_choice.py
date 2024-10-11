# ruff: noqa: E402
from prompting import settings

settings.settings = settings.Settings(mode="miner")
from prompting.rewards.multi_choice import MultiChoiceRewardModel
from dataclasses import dataclass
import pytest


@dataclass
class DendriteResponseEvent:
    completions: list[str]


JSON_PENALTY = 0.9

test_cases = [
    ("{\"A\": 0.1, \"B\": 0.3, \"C\": 0.6, \"D\": 0.0}", "C", 0.6),
    ("{\"A\": 0.1, \"B\": 0.3, \"C\": 0.6, \"D\": 0.0}", "A", 0.1),
    ("{\"a\": 0.0, \"b\": 0.0, \"c\": 1.0, \"d\": 0.0}", "C", 1.0),
    ("{\"a\": 0.0, \"b\": 0.0, \"c\": 1.0, \"d\": 0.0}", "D", 0),
    ("{\"A\": 0.1}", 'A', 1),
    ("{\"A\": 0.1, \"B\": 0.1}", "B", 0.5),
    ("{}", "A", 0),
    ("", "D", 0),
    ("Test", "A", 0),
    ("The answer is C.", "C", 1 * JSON_PENALTY),
    ("The answer is C or something like that", "C", 1 * JSON_PENALTY),
    ("The answer is D.", "C", 0),
]


@pytest.mark.parametrize("response, reference, expected", test_cases)
def test_logit_scoring(response, reference, expected):
    model = MultiChoiceRewardModel(json_penalty=JSON_PENALTY)
    result = model.reward(reference, DendriteResponseEvent(completions=[response])).rewards[0]
    assert result == pytest.approx(expected), f"Failed for input: {response}, reference: {reference}"
