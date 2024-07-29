from .summarization import SummarizationTask
from .qa import QuestionAnsweringTask
from .date_qa import DateQuestionAnsweringTask
from .task import Task  # noqa: F401

TASKS = {
    QuestionAnsweringTask.name: QuestionAnsweringTask,
    DateQuestionAnsweringTask.name: DateQuestionAnsweringTask,
    SummarizationTask.name: SummarizationTask,
}
