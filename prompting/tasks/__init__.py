from .task import Task
from .debugging import DebuggingTask
from .summarization import SummarizationTask
from .qa import QuestionAnsweringTask
from .date_qa import DateQuestionAnsweringTask
from .generic_instruction import GenericInstructionTask
from .math import MathTask
from .mock import MockTask


TASKS = {
    "mock": MockTask,
    "question-answering": QuestionAnsweringTask,
    "summarization": SummarizationTask,
    "date-based question answering": DateQuestionAnsweringTask,
    "debugging": DebuggingTask,
    "math": MathTask,
}
