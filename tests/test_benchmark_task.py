import sys
import pytest

from typing import Union
from types import SimpleNamespace
from itertools import permutations
from prompting.tasks import BenchmarkingTask
from prompting.validator import Validator
from prompting.utils.exceptions import TaskCreationError
from prompting.rewards import MultipleChoiceModel

from .fixtures.llm import mock_llm_pipeline
from .fixtures.dataset import MOCK_CONTEXT

QUESTION = 'What is the capital of Texas?'
OPTIONS = ['Austin','Houston','Paris','Bogota']

def make_multiple_choice_examples(question, options, correct_index, markers=None, base_marker=''):
    """Creates a multiple choice question (query) + answer (reference) pairs for different formatting variations of the question"""

    messages = []
    correct = options[correct_index]
    letters = ['A', 'B', 'C', 'D']
    formats = [
        '{letter}. {option}',
        '{letter}: {option}',
        '{letter}) {option}',
        '{letter}] {option}',
        '({letter}) {option}',
    ]
    markers = ['*','* ',' *'] if markers is None else markers

    for ordering in permutations(options):
        reference = letters[ordering.index(correct)]
        for form in formats:
            for marker in markers:
                for newlines in ['\n', '\n\n']:
                    query = f'{question}{newlines}'
                    for letter, option in zip(letters, ordering):
                        indicator = marker if option == correct else base_marker
                        query += f'{indicator}{form.format(letter=letter, option=option)}\n'
                    messages.append((query, reference))
    
    return messages

# we slice the (very large) lists to reduce the number of unnecessary tests while maintaining deterministic behaviour
good_query_examples = make_multiple_choice_examples(QUESTION, OPTIONS, correct_index=0)[::11]
missing_marker_examples = make_multiple_choice_examples(QUESTION, OPTIONS, correct_index=0, markers=[''])[::17]
multiple_marker_examples = make_multiple_choice_examples(QUESTION, OPTIONS, 0, base_marker='*')[::17]


@pytest.mark.parametrize(
    "message, reference", good_query_examples
)
def test_extract_query_and_reference_with_successful_generations(
    message: str, reference: str
):
    task = BenchmarkingTask(llm_pipeline=mock_llm_pipeline(message), context=MOCK_CONTEXT)
    assert task.query == "\n".join([line.strip("* ") for line in message.splitlines()])
    assert task.reference == reference
    
    
@pytest.mark.parametrize(
    "message, reference", missing_marker_examples
)
def test_extract_query_and_reference_with_missing_correct_answer_marker(
    message: str, reference: str
):
    with pytest.raises(TaskCreationError):
        task = BenchmarkingTask(llm_pipeline=mock_llm_pipeline(message), context=MOCK_CONTEXT)


@pytest.mark.parametrize(
    "message, reference", multiple_marker_examples
)
def test_extract_query_and_reference_with_multiple_correct_answer_markers(
    message: str, reference: str
):
    with pytest.raises(TaskCreationError):
        task = BenchmarkingTask(llm_pipeline=mock_llm_pipeline(message), context=MOCK_CONTEXT)


answers_templates = [
    'The correct answer is {reference}',
    'The correct answer: {reference}.',
    '{reference} is the correct answer.',
    '{reference} is correct because I saw the answer online.',
    'The most likely correct answer is {reference}.',
    "I don't know if it's A or B. But I'm going to just take a chance and guess {reference}..."
    'A, B are interesting ideas. Not sure about C. D is dumb. My final answer is {reference}.',    
]

@pytest.mark.parametrize(
    "message, reference", good_query_examples
)
@pytest.mark.parametrize(
    "answer_template", answers_templates
)
def test_correct_answer_scores_one(message, reference, answer_template):
    reward_model = MultipleChoiceModel()
    
    task = BenchmarkingTask(llm_pipeline=mock_llm_pipeline(message), context=MOCK_CONTEXT)
    
    # form a synthetic answer by filling in the template with the correct reference (e.g A)
    response = answer_template.format(reference=reference)
    response_event = SimpleNamespace(completions=[response])
    batch_reward_output = reward_model.reward(reference, response_event)
    
    assert all(reward == 1 for reward in batch_reward_output.rewards)
    
    
@pytest.mark.parametrize(
    "message, reference", good_query_examples
)
@pytest.mark.parametrize(
    "answer_template", answers_templates
)
def test_incorrect_answer_scores_zero(message, reference, answer_template):
    reward_model = MultipleChoiceModel()
    
    task = BenchmarkingTask(llm_pipeline=mock_llm_pipeline(message), context=MOCK_CONTEXT)
    
    # Choose the wrong answer
    wrong_answer = 'A' if reference!='A' else 'B'
    # form a synthetic answer by filling in the template with the correct reference (e.g A)
    response = answer_template.format(reference=wrong_answer)
    response_event = SimpleNamespace(completions=[response])
    batch_reward_output = reward_model.reward(reference, response_event)
    
    assert all(reward == 0 for reward in batch_reward_output.rewards)