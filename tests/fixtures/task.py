from prompting.tasks import Task, QuestionAnsweringTask, SummarizationTask, DebuggingTask, MathTask, DateQuestionAnsweringTask
from prompting.tools import MockDataset, CodingDataset, WikiDataset, StackOverflowDataset, DateQADataset, MathDataset
from fixtures.dataset import WIKI_ARTICLE

TASKS = [
        QuestionAnsweringTask,
        SummarizationTask,
        DebuggingTask,
        MathTask,
        DateQuestionAnsweringTask,
    ]

# TODO: Make fully deterministic
CONTEXTS = {
    QuestionAnsweringTask: WikiDataset().next(info=WIKI_ARTICLE),
    SummarizationTask: WikiDataset().next(info=WIKI_ARTICLE),
    DebuggingTask: CodingDataset(seed=123).next(),
    MathTask: MathDataset(seed=123).next(),
    DateQuestionAnsweringTask: DateQADataset().next(),
}

CONTEXTS = {
    QuestionAnsweringTask: {"text": "This is a context.", "title": "this is a title", "categories": ['some','categories']},
    SummarizationTask: {"text": "This is a context.", "title": "this is a title", "categories": ['some','categories']},
    DebuggingTask: {"code": "This is code","repo_name":'prompting',"path":'this/is/a/path', "language":'python'},
    MathTask: {"problem": "This is a problem","solution":'3.1415','topic':'Basic Algebra','subtopic':'Addition'},
    DateQuestionAnsweringTask: {"section": "Events", "event":"1953 - Battle of Hastings in UK", 'date':"1 January"},
}

CONTEXT = {"text": "This is a context.", "title": "this is a title"}
