import pytest
import time
import asyncio
import sys
from functools import partial
import bittensor as bt
from prompting.forward import run_step
from neurons.validator import Validator
from prompting.tasks import Task, QuestionAnsweringTask
from .fixtures.task import WIKI_CONTEXT
from prompting.agent import HumanAgent
from unittest.mock import patch, Mock
from prompting.protocol import PromptingSynapse, StreamPromptingSynapse

sys.argv = [__file__, "--mock", "--wandb.off", "--neuron.tasks", "qa"]
mock_neuron = Validator()

task = QuestionAnsweringTask(
    llm_pipeline=mock_neuron.llm_pipeline, context=WIKI_CONTEXT, create_reference=False
)


def generate_reference(x, delay=1):
    time.sleep(delay)
    return "Fake reference"


async def mock_dendrite_call(delay=1, **kwargs):
    time.sleep(delay)    
    
    async def async_fn_mock():                   
        mock_synapse = StreamPromptingSynapse(roles=["user"], messages=[""])
        mock_synapse.completion = "Fake response"
            
        yield mock_synapse
            
    mock_stream_synapse = async_fn_mock()                
    return [mock_stream_synapse]


@pytest.mark.parametrize(
    "generate_reference_time, dendrite_time, expected_forward_time",
    [(0.5, 0.5, 0.5), (0.5, 0.4, 0.5), (0.4, 0.5, 0.5)],
)
def test_generate_reference_parallel_to_dendrite(
    generate_reference_time, dendrite_time, expected_forward_time
):
    task.generate_reference = partial(generate_reference, delay=generate_reference_time)
    mock_agent = HumanAgent(task, mock_neuron.llm_pipeline)

    mock_neuron.dendrite = partial(mock_dendrite_call, delay=dendrite_time)

    event = asyncio.run(run_step(mock_neuron, mock_agent, k=4, timeout=0.1))

    step_time = event["step_time"]
    reward_pipeline_time = sum(
        event[key] for key in event if key.endswith("batch_time")
    )
    network_and_reference_gen_time = step_time - reward_pipeline_time

    assert network_and_reference_gen_time == pytest.approx(
        expected_forward_time, abs=0.1
    )
