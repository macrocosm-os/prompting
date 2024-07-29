from .tasks import (
    SummarizationTask,
    QuestionAnsweringTask,
    DateQuestionAnsweringTask,
)
from .tools import WikiDataset

TASK_REGISTRY = {
    SummarizationTask.name: [WikiDataset],
    QuestionAnsweringTask.name: [WikiDataset],
    DateQuestionAnsweringTask.name: [WikiDataset],
}
