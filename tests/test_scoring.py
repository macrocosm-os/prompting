import pytest
from datetime import datetime
from prompting.rewards import (
    DateRewardModel,
    DiffRewardModel,
    RelevanceRewardModel,
    RougeRewardModel,
    FloatDiffModel,
)
from prompting.rewards.pipeline import RewardPipeline

date1 = datetime.strptime("2022-01-01", "%Y-%m-%d")
date2 = datetime.strptime("2022-01-03", "%Y-%m-%d")
date3 = datetime.strptime("2020-01-08", "%Y-%m-%d")
date4 = datetime.strptime("2022-02-01", "%Y-%m-%d")
ref = "January 1, 2022"
date_formats = ["%B %d, %Y", "%m/%d/%Y", "%d %B %Y", "%m-%d-%Y"]
dates1 = [date1.strftime(format) for format in date_formats]
scores1 = [1.0] * len(dates1)
dates2 = [date2.strftime(format) for format in date_formats]
scores2 = [0.9960079893439915] * len(dates2)
dates3 = [date3.strftime(format) for format in date_formats]
scores3 = [0.0] * len(dates3)
dates4 = [date4.strftime(format) for format in date_formats]
scores4 = [0.38251018447178037] * len(dates4)
tuples = list(
    zip(dates1 + dates2 + dates3 + dates4, scores1 + scores2 + scores3 + scores4)
)


@pytest.mark.parametrize("reference", dates1)
@pytest.mark.parametrize("completion, expected_result", tuples)
def test_score_dates_with_different_format(reference, completion, expected_result):
    score = DateRewardModel().date_score(reference, completion)
    assert score == expected_result


completion = ["0.5", "1/2", "1-0.5", "2*0.25"]
expected_result = [1.0, 1.0, 1.0, 1.0]
reference = ["0.5"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing(reference, completion, expected_result):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result


completion = ["1e3", "-1e3", "1e-3", "-1e-3"]
expected_result = [1.0, 0.0, 0.0, 0.0]
reference = ["1000"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing_with_exponents(
    reference, completion, expected_result
):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result


completion = ["1.0.", "1.0", "1.0.0", "1,", "0 1"]
expected_result = [1.0, 1.0, 0.0, 1.0, 1.0]
reference = ["1.0"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing_with_punctuation(
    reference, completion, expected_result
):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result


completion = ["-20", "-23", "23", "20", "1000", "2*10+3"]
expected_result = [0.0, 0.0, 1.0, 0.8695652173918714, 0.0, 1.0]
reference = ["23"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing_with_negative_numbers(
    reference, completion, expected_result
):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result


completion = ["0", "0.001", "-0.0", "-0.001", "0.0001"]
expected_result = [1.0, 0.0, 1.0, 0.0, 0.0]
reference = ["0"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing_with_zeros(
    reference, completion, expected_result
):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result
