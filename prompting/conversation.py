
from prompting.tasks import DebuggingTask, QuestionAnsweringTask, SummarizationTask, MathTask, DateQuestionAnsweringTask
from prompting.tools import WikiDataset, CodingDataset, MathDataset, DateQADataset

def create_task(llm_pipeline, task_name):
    wiki_based_tasks = ['summarization', 'qa']
    coding_based_tasks = ['debugging']
    #TODO Add math and date_qa to this structure

    # TODO: Abstract dataset classes into common dynamic interface
    if task_name in wiki_based_tasks:
        dataset = WikiDataset()

    elif task_name in coding_based_tasks:
        dataset = CodingDataset()

    elif task_name == 'math':
        dataset = MathDataset()

    elif task_name == 'date_qa':
        dataset = DateQADataset()


    if task_name == 'summarization':
        return SummarizationTask(llm_pipeline=llm_pipeline, context=dataset.next())

    elif task_name == 'qa':
        return QuestionAnsweringTask(llm_pipeline=llm_pipeline, context=dataset.next())

    elif task_name == 'debugging':
        return DebuggingTask(llm_pipeline=llm_pipeline, context=dataset.next())

    elif task_name == 'math':
        return MathTask(llm_pipeline=llm_pipeline, context=dataset.next())

    elif task_name == 'date_qa':
        return DateQuestionAnsweringTask(llm_pipeline=llm_pipeline, context=dataset.next())

    else:
        raise ValueError(f'Task {task_name} not supported. Please choose a valid task')
