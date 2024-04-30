from prompting.tasks import Task

QUERY_PROMPT_TEMPLATE = """\
You are a review-generating expert, focusing on making highly reaslistic revies. Your response contains only the review, nothing more, nothing less. You will adhere to a word limit of 250 words. Ask a specific question about the following context:
{context}
"""


class SentimentAnalysisTask(Task):
    important_field = ["{context"]
    name = "sentiment"
    desc = "get help analyzing the sentiment of a review"
    goal = "to get the sentiment to the following review"
    challenge_type = 'paraphrase'
    challenge_template = "What is the sentiment of the following review: {context}"


    reward_definition = [
        dict(name="ordinal", weight=1.0),
    ]
    penalty_definition = []
    cleaning_pipeline = []

    static_reference = True

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        self.query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.content)
        self.query = self.generate_query(llm_pipeline)
        self.reference = context.subtopic

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    def format_challenge(self, challenge) -> str:
        return challenge.format(context = self.query)