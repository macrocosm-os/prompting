
class DebuggingTask:
    reward_models = [
        dict(name='diff', lines=False, threshold=0.5),
        dict(name='relevance', threshold=None),
    ]
    
class SummarizationTask:
    reward_models = [
        dict(name='rouge', ngram='rouge-l', metric='f'),
        dict(name='relevance', threshold=None),
    ]    

class QuestionAnsweringTask:
    reward_models = [
        dict(name='rouge', ngram='rouge-1', metric='f'),
        dict(name='relevance', threshold=None),
    ]    

class MathTask:
    reward_models = [
        dict(name='rouge', ngram='rouge-l', metric='f'),
    ]
    
class DateQuestionAnsweringTask:
    reward_models = [
        dict(name='rouge', ngram='rouge-l', metric='f'),
    ]
