import pytest
import json
from prompting.rewards.multi_choice import MultiChoiceRewardModel

invalid_responses = ["This is a text string", 'This is a semi JSON response {"A":0.1}', '{"a":"1.0"}']
incomplete_responses = ["{}", '{"A":0.1}', '{"A":0.1, "B":0.9}']
unnormalized_responses = [
    '{"a":1.23}',
    '{"A":0.1, "B":0.1}',
    '{"A":0.1, "B":0.9, "C":0.0}',
    '{"a":0,"b":1,"c":0,"d":1}',
]

completions = [(text, lambda x: 0) for text in invalid_responses + incomplete_responses + unnormalized_responses]

valid_responses = [
    '{"a":1.0}',
    '{"A":0.1, "B":0.3, "C":0.6}',
    '{"A":0.1, "B":0.3, "C":0.6, "D":0.0}',
    '{"a":0.0, "b":0.0, "c":1.0, "d":0.0}',
]
multiline_responses = [text.replace(",", ",\n") for text in valid_responses]
multiline_responses += [text.replace("}", ",\n}") for text in valid_responses]

completions = [(text, lambda x: json.loads(text)[x]) for text in valid_responses + valid_responses]


@pytest.mark.parametrize("reference", MultiChoiceRewardModel.choices)
@pytest.mark.parametrize("completion, expected_result", completions)
def test_logit_scoring(reference: str, completion: str, expected_result: float):
    score = MultiChoiceRewardModel()._logit_reward(reference, completion)

    assert score == expected_result
