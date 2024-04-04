from prompting.tasks import (
    Task,
    DebuggingTask,
    QuestionAnsweringTask,
    SummarizationTask,
    MathTask,
    DateQuestionAnsweringTask,
)
from prompting.tools import (
    WikiDataset,
    HFCodingDataset,
    MathDataset,
    WikiDateDataset,
)

from transformers import Pipeline


def create_task(llm_pipeline: Pipeline, task_name: str, create_reference=True) -> Task:
    wiki_based_tasks = ["summarization", "qa"]
    coding_based_tasks = ["debugging"]
    # TODO: Abstract dataset classes into common dynamic interface
    if task_name in wiki_based_tasks:
        dataset = WikiDataset()

    elif task_name in coding_based_tasks:
        dataset = HFCodingDataset()

    elif task_name == "math":
        dataset = MathDataset()

    elif task_name == "date_qa":
        dataset = WikiDateDataset()

    if task_name == "summarization":
        task = SummarizationTask(
            llm_pipeline=llm_pipeline,
            context=dataset.next(),
            create_reference=create_reference,
        )

    elif task_name == "qa":
        task = QuestionAnsweringTask(
            llm_pipeline=llm_pipeline,
            context=dataset.next(),
            create_reference=create_reference,
        )

    elif task_name == "debugging":
        task = DebuggingTask(
            llm_pipeline=llm_pipeline,
            context=dataset.next(),
            create_reference=create_reference,
        )

    elif task_name == "math":
        task = MathTask(
            llm_pipeline=llm_pipeline,
            context=dataset.next(),
            create_reference=create_reference,
        )

    elif task_name == "date_qa":
        task = DateQuestionAnsweringTask(
            llm_pipeline=llm_pipeline,
            context=dataset.next(),
            create_reference=create_reference,
        )

    else:
        raise ValueError(f"Task {task_name} not supported. Please choose a valid task")

    return task
