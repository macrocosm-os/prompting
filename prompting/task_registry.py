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
)

# TODO: Expand this to include extra information beyond just the task and dataset names
summarization_task, summarization_dataset = SummarizationTask.name, [WikiDataset.name]
qa_task, qa_dataset = QuestionAnsweringTask.name, [WikiDataset.name]
benchmark_task, benchmark_dataset = BenchmarkingTask.name, [WikiDataset.name]

# debugging_task, debugging_dataset = DebuggingTask.name, [HFCodingDataset.name]
math_task, math_dataset = MathTask.name, [MathDataset.name]
date_qa_task, date_qa_dataset = DateQuestionAnsweringTask.name, [WikiDateDataset.name]
generic_instruction_task, generic_instruction_dataset = GenericInstructionTask.name, [
    GenericInstructionDataset.name
]
translation_task, translation_dataset = TranslationTask.name, [WikiDataset.name]
sentiment_analysis_task, sentiment_analysis_dataset = SentimentAnalysisTask.name, [
    ReviewDataset.name
]

TASK_REGISTRY = {
    summarization_task: summarization_dataset,
    qa_task: qa_dataset,
    # debugging_task: debugging_dataset,
    math_task: math_dataset,
    date_qa_task: date_qa_dataset,
    generic_instruction_task: generic_instruction_dataset,
    translation_task: translation_dataset,
    sentiment_analysis_task: sentiment_analysis_dataset,
    benchmark_task: benchmark_dataset,
}
