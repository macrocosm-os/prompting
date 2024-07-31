# import random
# from transformers import Pipeline
# from prompting.tasks.task import BaseTask
# from prompting.tools import Selector
# from prompting.task_registry import TaskRegistry, TaskConfig
# from prompting.tools.datasets.base import BaseDataset


# def create_task(
#     llm_pipeline: Pipeline,
#     task_config: TaskConfig,
#     create_reference: bool = True,
#     selector: Selector = random.choice,
# ) -> BaseTask:
#     """Create a task from the given task name and LLM pipeline.

#     Args:
#         llm_pipeline (Pipeline): Pipeline to use for text generation
#         task_config (str): TaskConfig object to create task from
#         create_reference (bool, optional): Generate text for task reference answer upon creation. Defaults to True.
#         selector (Selector, optional): Selector function to choose a dataset. Defaults to random.choice.

#     Raises:
#         ValueError: If task_name is not a valid alias for a task, or if the task is not a subclass of Task
#         ValueError: If no datasets are available for the given task
#         ValueError: If the dataset for the given task is not found

#     Returns:
#         Task: Task instance
#     """
#     dataset: BaseDataset = selector(task_config.datasets)()

#     return task_config.task(
#         llm_pipeline=llm_pipeline,
#         context=dataset.next(),
#         create_reference=create_reference,
#     )
