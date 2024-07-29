from .tasks import (
    SummarizationTask,
    QuestionAnsweringTask,
    DateQuestionAnsweringTask,
    TranslationTask,
)
from .tools import WikiDataset, WikiDateDataset

# TODO: Expand this to include extra information beyond just the task and dataset names
summarization_task, summarization_dataset = SummarizationTask.name, [WikiDataset.name]
qa_task, qa_dataset = QuestionAnsweringTask.name, [WikiDataset.name]
# debugging_task, debugging_dataset = DebuggingTask.name, [HFCodingDataset.name]
date_qa_task, date_qa_dataset = DateQuestionAnsweringTask.name, [WikiDateDataset.name]
translation_task, translation_dataset = TranslationTask.name, [WikiDataset.name]

TASK_REGISTRY = {
    summarization_task: summarization_dataset,
    qa_task: qa_dataset,
    date_qa_task: date_qa_dataset,
}
