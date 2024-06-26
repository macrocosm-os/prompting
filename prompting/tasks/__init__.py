from .task import Task
from .debugging import DebuggingTask
from .summarization import SummarizationTask
from .qa import QuestionAnsweringTask
from .date_qa import DateQuestionAnsweringTask
from .generic_instruction import GenericInstructionTask
from .math import MathTask
from .translate import TranslationTask, TranslationPipeline
from .mock import MockTask
from .sentiment import SentimentAnalysisTask
from .benchmark import BenchmarkingTask


TASKS = {
    QuestionAnsweringTask.name: QuestionAnsweringTask,
    DateQuestionAnsweringTask.name: DateQuestionAnsweringTask,
    SummarizationTask.name: SummarizationTask,
    # DebuggingTask.name: DebuggingTask,
    GenericInstructionTask.name: GenericInstructionTask,
    MathTask.name: MathTask,
    TranslationTask.name: TranslationTask,
    SentimentAnalysisTask.name: SentimentAnalysisTask,
    BenchmarkingTask.name: BenchmarkingTask,
}
