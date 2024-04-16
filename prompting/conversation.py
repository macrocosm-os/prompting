import random
from transformers import Pipeline
from prompting.tasks import Task, TASKS
from prompting.tools import Selector, DATASETS
from prompting.task_registry import TASK_REGISTRY


def random_task(self, include=None, exclude=None, history=None):
    choices = include or [task_name for task_name in self.config.neuron.tasks if not type(exclude)==list or task_name not in exclude]
    assert choices, f'No valid choices for {include=}, {exclude=}'
    probs = torch.FloatTensor([self.config.neuron.task_p.index(choice) for choice in choices])
    probs /= probs.sum()
    
    while True:
        bt.logging.info(
            f"📋 Selecting task... from {choices=} with distribution {probs.tolist()=}"
        )
        # Create a specific task
        task_name = np.random.choice(choices, probs.tolist())
        bt.logging.info(f"📋 Creating {task_name} task... ")
        try:
            return create_task(
                llm_pipeline=self.llm_pipeline,
                task_name=task_name,
                create_reference=False,
                history=None,
            )
        except Exception as e:
            bt.logging.error(
                f"Failed to create {task_name} task. {sys.exc_info()}. Skipping to next task."
            )

def create_task(
    llm_pipeline: Pipeline,
    task_name: str,
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

    task = TASKS.get(task_name, None)
    if task is None or not issubclass(task, Task):
        raise ValueError(f"Task {task_name} not found")

    dataset_choices = TASK_REGISTRY.get(task_name, None)
    if len(dataset_choices) == 0:
        raise ValueError(f"No datasets available for task {task_name}")

    dataset_name = selector(dataset_choices)
    dataset = DATASETS.get(dataset_name, None)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} not found")
    else:
        dataset = dataset()

    return task(
        llm_pipeline=llm_pipeline,
        context=dataset.next(),
        create_reference=create_reference,
    )