from prompting.tasks import Task
from prompting.cleaners.cleaner import CleanerPipeline


class SentimentAnalysisTask(Task):
    name = "sentiment analysis"
    desc = "get help analyzing the sentiment of a review"
    goal = "to get the sentiment to the following review"
    reward_definition = [
        dict(name="sentiment", weight=1.0),
    ]
    penalty_definition = []
    cleaning_pipeline = []

    static_reference = True
    query_system_prompt = """You are an assistant that generates reviews based on user prompts. You follow all of the user instructions as well as you can. Make the reviews as realistic as possible. Your response contains only the review, nothing more, nothing less"""

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        self.query_prompt = context.content
        self.query = self.generate_query(llm_pipeline)
        self.reference = context.subtopic

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
