import random
from transformers import Pipeline
from prompting.tasks import Task
from prompting.tools import Selector
from prompting.task_registry import TASK_REGISTRY
from prompting.tools.datasets.base import BaseDataset


def create_task(
    llm_pipeline: Pipeline,
    task: Task,
    create_reference: bool = True,
    selector: Selector = random.choice,
) -> Task:
    """Create a task from the given task name and LLM pipeline.

    Args:
        llm_pipeline (Pipeline): Pipeline to use for text generation
        task_name (str): Name of the task to create
        create_reference (bool, optional): Generate text for task reference answer upon creation. Defaults to True.
        selector (Selector, optional): Selector function to choose a dataset. Defaults to random.choice.

    Raises:
        ValueError: If task_name is not a valid alias for a task, or if the task is not a subclass of Task
        ValueError: If no datasets are available for the given task
        ValueError: If the dataset for the given task is not found

    Returns:
        Task: Task instance
    """

    dataset_choices = TASK_REGISTRY.get(task.name)

    if len(dataset_choices) == 0:
        raise ValueError(f"No datasets available for task {task.name}")

    dataset: BaseDataset = selector(dataset_choices)()

    return task(
        llm_pipeline=llm_pipeline,
        context=dataset.next(),
        create_reference=create_reference,
    )
