import pytest
from unittest.mock import MagicMock

import bittensor as bt

from prompting.tasks import (
    QuestionAnsweringTask,
    SummarizationTask,
)


@pytest.fixture
def setup():
    # Ensure that anytime the pipeline is called, it returns a MagicMock infinitely.
    mock_llm_pipeline = MagicMock()
    mock_llm_pipeline.return_value = MagicMock()
    mock_context = MagicMock()
    create_reference = True
    return mock_llm_pipeline, mock_context, create_reference


def test_example(setup):
    mock_llm_pipeline, mock_context, create_reference = setup

    for task in [
        QuestionAnsweringTask,
        SummarizationTask,
    ]:
        bt.logging.info(f"Testing task {task}...")

        task(
            llm_pipeline=mock_llm_pipeline,
            context=mock_context,
            create_reference=create_reference,
        )
