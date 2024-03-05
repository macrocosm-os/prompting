# llm, input, expected output, cleaner

##test_llm_forward
# test llm query (check messages, times)
# test llm query (calls forward, clean_response)

import pytest
from prompting.llms import BaseLLM, BasePipeline
from prompting.cleaners import CleanerPipeline
from prompting.mock import MockPipeline
from .fixtures.llm import llms, pipelines
from .fixtures.cleaner import DEFAULT_CLEANER_PIPELINE


@pytest.mark.parametrize(
    "input, expected_result, cleaner",
    [
        (
            '"I am a quote. User: I know you are. I am asking a question. What is th"',
            '"I am a quote. User: I know you are. I am asking a question. What is th"',
            None,
        ),
        (
            '"I am a quote. User: I know you are. I am asking a question. What is th"',
            "I am a quote. I know you are. I am asking a question.",
            DEFAULT_CLEANER_PIPELINE,
        ),
    ],
)
@pytest.mark.parametrize("llm", llms())
def test_llm_clean_response(
    input: str, expected_result: str, cleaner: CleanerPipeline, llm: BaseLLM
):
    result = llm.clean_response(cleaner=cleaner, response=input)
    assert result == expected_result


@pytest.mark.parametrize("pipeline", pipelines())
def test_load_pipeline_mock(pipeline: BasePipeline):
    # Note that the model_id will be used internally as static response for the mock pipeline
    model_id = "gpt2"
    pipeline_instance = pipeline(model_id=model_id, device="cpu", mock=True)
    pipeline_message = pipeline_instance("")

    mock_message = MockPipeline(model_id).forward(messages=[])
    assert mock_message == pipeline_message


@pytest.mark.parametrize("llm", llms())
def test_llm_query(llm: BaseLLM):
    message = "test"
    llm.query(message)

    # Assert that stateful operation where 3 messages are saved:
    # the system prompt (on llm init), the user message and the assistant reply
    assert len(llm.messages) == 3
    assert len(llm.times) == 3

    assert llm.messages[0]["role"] == "system"

    assert llm.messages[1]["role"] == "user"
    assert llm.messages[1]["content"] == message

    assert llm.messages[2]["role"] == "assistant"


@pytest.mark.parametrize("llm", llms())
def test_llm_forward(llm: BaseLLM):
    llm.forward(llm.messages)

    # Assert stateless operation of the model with only history of system prompt
    assert len(llm.messages) == 1
    assert len(llm.times) == 1
    assert llm.messages[0]["role"] == "system"
