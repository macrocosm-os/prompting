from .task import Task
from .debugging import DebuggingTask
from .summarization import SummarizationTask
from .qa import QuestionAnsweringTask
from .date_qa import DateQuestionAnsweringTask
from .generic_instruction import GenericInstructionTask
from .math import MathTask
from .mock import MockTask


TASKS = {
    QuestionAnsweringTask.name: QuestionAnsweringTask,
    DateQuestionAnsweringTask.name: DateQuestionAnsweringTask,
    SummarizationTask.name: SummarizationTask,
    # DebuggingTask.name: DebuggingTask,
    GenericInstructionTask.name: GenericInstructionTask,
    MathTask.name: MathTask,
}
