from .tasks import Task, MockTask, SummarizationTask, QuestionAnsweringTask, DebuggingTask, MathTask, DateQuestionAnsweringTask
from .tools import MockDataset, WikiDataset, HFCodingDataset, StackOverflowDataset, MathDataset, WikiDateDataset

# TODO: Expand this to include extra information beyond just the task and dataset names
mock_task, mock_dataset = MockTask.name, [MockDataset.name]
summarization_task, summarization_dataset = SummarizationTask.name, [WikiDataset.name]
qa_task, qa_dataset = QuestionAnsweringTask.name, [WikiDataset.name]
debugging_task, debugging_dataset = DebuggingTask.name, [HFCodingDataset.name]
math_task, math_dataset = MathTask.name, [MathDataset.name]
date_qa_task, date_qa_dataset = DateQuestionAnsweringTask.name, [WikiDateDataset.name]

TASK_REGISTRY = {
    mock_task: mock_dataset,
    summarization_task: summarization_dataset,
    qa_task: qa_dataset,
    debugging_task: debugging_dataset,
    math_task: math_dataset,
    date_qa_task: date_qa_dataset,
}