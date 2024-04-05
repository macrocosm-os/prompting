from prompting.tasks import Task


class MathTask(Task):
    name = "math"
    desc = "get help solving a math problem"
    goal = "to get the answer to the following math problem"

    reward_definition = [
        dict(name="float_diff", weight=1.0),
    ]
    penalty_definition = []

    static_reference = True
    static_query = True

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        self.query = (
            "How can I solve the following problem, "
            + context.content
            + "? You must include the whole problem in your question."
        )
        self.reference = context.extra["solution"]
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
