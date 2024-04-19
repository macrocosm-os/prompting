import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments
# TODO

# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. The questions you generate should be based on the context that is provided.
You will maintain a neutral tone in your questions.
You will adhere to a word limit of 50 words for each question.
"""

# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context:

#Context:
{context}
"""

# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Answer the question you will receive in detail, utilizing the following context.

#Context:
{context}

# Question:
{question}
"""

# Used to obtain the query (which is a followup question about the context)
FOLLOWUP_PROMPT_TEMPLATE = """
Ask a specific question to continue the conversation below. You must adopt the same persona as the human user (tone, style, goals). It must be possible to answer your followup question objectively, but you must not answer it. You may use using the provided context as the basis for the followup question, but it is not a requirement. The assistant does not have direct access to the context, so you should not refer to it directly.

Importantly, your followup question must require the conversation history to answer correctly. This can be achieved by using implicit and indirect language (can tell me more about that? what was the reason ...? why did she ...?), or if you are confident that the assistant response was wrong or not useful you can request further information or point out any problems you encounter. You must not answer your own question. If the original user query was itself of poor quality you may use the followup question to clarify and amend it. It can be based on any message in the conversation history.

# Context:
{context}

# Conversation History:
{history}
"""
# TODO: We also need a special followup reference prompt (or just merge both)
# Used to obtain reference answer
FOLLOWUP_REFERENCE_PROMPT_TEMPLATE = """\
Answer the question you will receive in detail, utilizing the following context and conversation history as required. 

#Context:
{context}

# Conversation History:
{history}

# Question:
{question}
"""

@dataclass
class QuestionAnsweringTask(Task):
    name = "qa"
    desc = "get help on answering a question"
    goal = "to get the answer to the following question"

    reward_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
        dict(name="relevance", weight=0.5),
    ]
    penalty_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
    ]

    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
    ]

    def __init__(self, llm_pipeline, context, create_reference=True, history=None):
        self.context = context

        self.query_system_prompt = QUERY_SYSTEM_PROMPT
        if history:
            self.query_prompt = FOLLOWUP_PROMPT_TEMPLATE.format(context=context.content, history=history)
            bt.logging.warning(f'Using history!!\n{history=}\n\n{context=}\n\n{self.query_prompt=}')
        else:
            self.query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.content)            
            
        self.query = self.generate_query(llm_pipeline)

        if history:
            self.reference_prompt = FOLLOWUP_REFERENCE_PROMPT_TEMPLATE.format(
                context=context.content, question=self.query, history=history
            )
        else:
            self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(
                context=context.content, question=self.query
            )            
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)
        
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
