from prompting.tasks import Task, QuestionAnsweringTask, SummarizationTask, DebuggingTask, MathTask, DateQuestionAnsweringTask
from .dataset import WIKI_CONTEXT, CODING_CONTEXT, MATH_CONTEXT, DATEQA_CONTEXT

TASKS = [
        QuestionAnsweringTask,
        SummarizationTask,
        DebuggingTask,
        MathTask,
        DateQuestionAnsweringTask,
    ]

# TODO: Make fully deterministic
CONTEXTS = {
    QuestionAnsweringTask: WIKI_CONTEXT,
    SummarizationTask: WIKI_CONTEXT,
    DebuggingTask: CODING_CONTEXT,
    MathTask:  MATH_CONTEXT,
    DateQuestionAnsweringTask: DATEQA_CONTEXT,
}

