from prompting.tasks import Task

# TODO: (this doesn't belong here but...) We can measure expressivity of an LLM(pipeline) by creating MANY challenges with the same prompt and measuring the variety (superificial and semantic) of the completions. We have already seen that zephyr is limited in this way: even with different personas there is a lot of repetition.

PARAPHRASE_TEMPLATE = """\

How do I translate the following text from {from_lang} to {to_lang}?

{text}
"""


class TranslationTask(Task):
    name = "translation"
    desc = "get help translating text"
    goal = "to translate the following text"

    reward_definition = [
        dict(name="rouge", weight=1.0),
    ]
    penalty_definition = []

    static_reference = True
    static_query = True
    paraphrase_query = True

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        self.query = context.content
        self.reference = context.extra["translation"]

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    def load_packages(self):
        ...
