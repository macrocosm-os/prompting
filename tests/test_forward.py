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

@pytest.mark.parametrize('delay', [0.25, 0.5, 1])
@pytest.mark.parametrize('min_time', [0.25, 0.5, 1])
@pytest.mark.parametrize('trial', range(1))
def test_generate_reference_while_waiting_for_dendrite(delay, min_time, trial):

    # force the mock dendrite to take at least min_time to respond
    mock_neuron.dendrite.min_time = min_time
    task.generate_reference = partial(generate_reference, delay=delay)
    agent = HumanAgent(task, mock_neuron.llm_pipeline)

    # TODO: check if dendrites have status codes 200 or 408 (which indicates that task reference was blocking)
    timeout = 1
    async def run():
        return await run_step(mock_neuron, agent, k=4, timeout=timeout)

    event = asyncio.run(run())
    bt.logging.info(event)

    step_time = event['step_time']
    eps = 0.1
    assert step_time < max(delay, mock_neuron.dendrite.max_time) + eps, "Timeout not respected."
    