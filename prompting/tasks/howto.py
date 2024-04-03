from prompting.tasks import Task


class HowToTask(Task):
    name = "how to"
    desc = "get help translating text"
    goal = "to translate the following text"

    reward_definition = [
        dict(name="rouge", weight=1.0),
    ]
    penalty_definition = []

    static_reference = True
    static_query = True

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        self.query = f"How do I {context.title}?"
        self.reference = context.content

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
