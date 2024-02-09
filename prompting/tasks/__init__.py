from .task import Task
from .debugging import DebuggingTask
from .summarization import SummarizationTask
from .qa import QuestionAnsweringTask
from .date_qa import DateQuestionAnsweringTask
from .generic_instruction import GenericInstructionTask
from .math import MathTask


TASKS = {
    "debugging": DebuggingTask,
    "summarization": SummarizationTask,
    "qa": QuestionAnsweringTask,
    "math": MathTask,
    "date_qa": DateQuestionAnsweringTask,
}
