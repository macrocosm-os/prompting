import pytest
import torch
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

sys.argv = [__file__, '--mock', '--wandb.off','--neuron.tasks','qa']
mock_neuron = Validator()

task = QuestionAnsweringTask(llm_pipeline=mock_neuron.llm_pipeline, context=WIKI_CONTEXT, create_reference=False)

def generate_reference(x, delay=1):
    time.sleep(delay)
    return 'Fake reference'

@pytest.mark.parametrize('delay', [0.1, 0.2, 0.3])
@pytest.mark.parametrize('timeout', [0.1, 0.2])
@pytest.mark.parametrize('min_time', [0, 0.05, 0.1])
@pytest.mark.parametrize('max_time', [0.1, 0.15, 0.2])
def test_generate_reference_while_waiting_for_dendrite(delay, timeout, min_time, max_time):

    # force the mock dendrite to take at least min_time to respond
    mock_neuron.dendrite.min_time = min_time
    mock_neuron.dendrite.max_time = max_time
    task.generate_reference = partial(generate_reference, delay=delay)
    agent = HumanAgent(task, mock_neuron.llm_pipeline)

    async def run():
        return await run_step(mock_neuron, agent, k=4, timeout=timeout)

    event = asyncio.run(run())

    step_time = event['step_time']
    network_time = step_time - sum(event[key] for key in event if key.endswith('batch_time'))
    eps = 0.1
    assert network_time <= max(delay, mock_neuron.dendrite.max_time) + eps, "Timeout not respected."

    # check that even when delay > timeout, the dendrites still have 200 status codes
    # if delay > timeout:
    #     assert all(status_code == 408 for status_code in event['status_codes'])
