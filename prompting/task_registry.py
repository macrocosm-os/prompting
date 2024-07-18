from .tasks import (
    Task,
    MockTask,
    SummarizationTask,
    QuestionAnsweringTask,
    DebuggingTask,
    MathTask,
    DateQuestionAnsweringTask,
    GenericInstructionTask,
    SentimentAnalysisTask,
    TranslationTask,
    BenchmarkingTask,
)
from .tools import (
    MockDataset,
    WikiDataset,
    HFCodingDataset,
    StackOverflowDataset,
    MathDataset,
    WikiDateDataset,
    GenericInstructionDataset,
    ReviewDataset,
    ArxivDataset,
)

# TODO: Expand this to include extra information beyond just the task and dataset names
TASK_REGISTRY = {
    SummarizationTask.name: [WikiDataset.name, ArxivDataset.name],
    QuestionAnsweringTask.name: [WikiDataset.name, ArxivDataset.name],
    BenchmarkingTask.name: [WikiDataset.name, ArxivDataset.name],
    # debugging_task: debugging_dataset,
    MathTask.name: [MathDataset.name],
    DateQuestionAnsweringTask.name: [WikiDateDataset.name, ArxivDataset.name],
    GenericInstructionTask.name: [GenericInstructionDataset.name],
    TranslationTask.name: [WikiDataset.name],
    SentimentAnalysisTask.name: [ReviewDataset.name],
}
