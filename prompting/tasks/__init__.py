from .task import Task, CHATTENSOR_SYSTEM_PROMPT
from .debugging import DebuggingTask
from .summarization import SummarizationTask
from .qa import QuestionAnsweringTask
from .date_qa import DateQuestionAnsweringTask
from .generic import GenericInstructionTask
from .math import MathTask
from .sentiment import SentimentAnalysisTask
from .generic import GenericInstructionTask
from .translate import TranslationTask
from .howto import HowToTask

TASKS = {
    "qa": QuestionAnsweringTask,
    "summarization": SummarizationTask,
    "date_qa": DateQuestionAnsweringTask,
    # "debugging": DebuggingTask,
    "math": MathTask,
    "sentiment": SentimentAnalysisTask,
    "generic": GenericInstructionTask,
    "translation": TranslationTask,
    "howto": HowToTask,
}
