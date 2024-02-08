import pytest
import torch
import time
import asyncio
import sys
from prompting.forward import run_step
from neurons.validator import Validator
from prompting.tasks import Task, QuestionAnsweringTask
from .fixtures.task import WIKI_CONTEXT
from prompting.agent import HumanAgent

sys.argv = [__file__,'--mock', '--wandb.off']
mock_neuron = Validator()


def generate_reference(x):
    time.sleep(1)
    return 'Fake reference'


task = QuestionAnsweringTask(llm_pipeline=mock_neuron.llm_pipeline, context=WIKI_CONTEXT, create_reference=False)
task.generate_reference = generate_reference


def test_generate_reference_while_waiting_for_dendrite():

    agent = HumanAgent(task, mock_neuron.llm_pipeline)
    t0 = time.time()
    async def run():
        await run_step(mock_neuron, agent, k=2, timeout=1)

    asyncio.run(run())
    assert time.time() - t0 < 2, "Timeout not respected."